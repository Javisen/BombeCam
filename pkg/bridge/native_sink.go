package bridge

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// nativeSink turns the camera's RTP media into RTSP publications on MediaMTX
// without FFmpeg: it rebuilds NAL units, groups them into access units by RTP
// timestamp, derives stable timestamps, and (re)connects the publisher.
//
// Timestamps: every published track has strictly increasing timestamps, no
// matter what the network or the camera does. Recorders (Frigate's FFmpeg
// segmenter) and MediaMTX's HLS muxer reject or restart on a timestamp that
// goes backwards. Video follows the camera's own frame spacing, gently steered
// toward the wall clock; a stall followed by a burst of late frames is absorbed
// by the camera clock instead of being stamped with arrival times. Audio
// advances by one AAC frame (1024 samples) per frame, and counts frames the
// camera's RTP clock says were lost.
//
// Audio/video alignment: MediaMTX lines tracks up by when their packets
// arrive (the first packet of every track, and all packets for WebRTC
// viewers, whose sender reports use send time). Audio is naturally "older"
// than video when it arrives (an AAC frame can only be sent once all of its
// samples were captured, and the browser copy also passes through a
// transcoder), so when a stream has sound every track is handed to MediaMTX
// at (capture time + one shared delay). The delay adapts to the largest
// observed lateness (typically 0.1-0.3 s) and shrinks back when tracks have
// been on time for a while; streams without sound are sent immediately.
type nativeSink struct {
	rtspURL   string
	label     string
	wantAudio bool
	now       func() time.Time                  // injectable clock (tests)
	auHook    func(camTS uint32, nals [][]byte) // tests: every access unit flushed

	mu       sync.Mutex
	closed   bool
	pub      *rtspPublisher
	lastDial time.Time
	dialErr  string
	start    time.Time

	asm      *nalAssembler
	haveAU   bool
	auTS     uint32
	auWall   time.Time
	auNALs   [][]byte
	sps, pps []byte
	sawIDR   bool
	firstPkt time.Time

	// late packets from an access unit that was already flushed
	haveLastAU  bool
	lastAUTS    uint32
	oldPktRun   int
	droppedLate int64

	// video timestamp mapping (90 kHz ticks since start)
	vAnchored bool
	vBase     int64
	vElapsed  int64
	vLastCam  uint32
	vStep     int64 // last plausible frame interval (ticks)
	vLastOut  int64

	// stream description (S9)
	spsInfo  h264SPS
	haveInfo bool
	fpsWin   time.Time
	fpsCount int
	fps      float64

	// audio (sample-rate ticks since start)
	audioCfg      *aacConfig
	aFramesPerPkt int
	aAnchored     bool
	aNext         int64 // timeline position of the next AAC frame
	aHaveRTP      bool
	aLastRTP      uint32
	aLastFrames   int
	aRTPStep      uint32 // camera RTP ticks per AAC frame (learned)
	aStepCand     uint32
	aStepVotes    int
	aClock        uint32 // audio RTP clock rate (0 = the AAC sample rate)
	droppedAudio  int64
	g711Next      int64

	// How much real time one camera AAC frame covers. The WS03 sends a frame
	// every 120 ms and steps its RTP clock 960 per frame, although decoders
	// turn each frame into 1024 samples (128 ms). The timeline must follow
	// real time, or audio runs ahead of the video by ~67 ms every second.
	// The RTP step is used, checked against the arrival cadence (aMeas*).
	aFrameSamples int
	aMeasStart    time.Time
	aMeasRTP      uint32
	aMeasLast     time.Time
	aMeasCand     int
	aMeasVotes    int

	// Audio that runs ahead of its arrival time (a start-up backlog, or a
	// late first packet the timeline was anchored on) would be delivered
	// that much later for the whole session. aWin* watch the smallest lead
	// over a window; persistent lead is removed by skipping stale frames.
	aWinStart   time.Time
	aWinMinLead int64
	aWinMaxLead int64
	aWinHave    bool
	aDropBudget int
	leadDrops   int64
	leadGaps    int64

	g711Gen int // generation of the current transcoder (restarts ignore stale output)

	// timing summary logged once a minute (diagnostics for latency reports)
	stAt                   time.Time
	stALead, stVLag, stG   [2]int64 // min, max in ms
	stHaveA, stHaveV, stHG bool
	stGrowTrack            int
	stGrowAge              time.Duration

	published int64

	// FFmpeg path for the optional G.711 browser-audio track ("" = none).
	transcoderPath string
	g711           *g711Transcoder

	// delivery alignment (see type comment)
	delay    time.Duration
	minDelay time.Duration

	// Last line of defence at write time (timestamps.go): each track's
	// published timestamps are forced to increase. The clocks above already
	// never go backwards; this only catches what they might miss.
	guard       [3]tsMonotonic
	guardNote   [3]bool      // a hold was logged for this publication
	guardHeld   int          // timestamps the guard had to hold (tests, diagnostics)
	lateSince   [3]time.Time // start of the current run of late items, per track
	lateDropped int64        // stale browser-audio packets dropped (can't be in sync)
	winStart    time.Time
	winMaxAge   time.Duration
	decayTarget time.Duration
	lastGlide   time.Time
	queues      [3][]sinkItem
	wake        chan struct{}
	scheduler   bool

	// Viewer copy: the video alone, published at once on arrival to
	// <path>/viewer, for the page's muted live view. The main publication
	// holds video back by the audio/video alignment delay; a picture without
	// sound doesn't need that.
	vpub        *rtspPublisher
	vpubDialing bool
	vpubLast    time.Time
	vpubSawIDR  bool
	vpubGuard   tsMonotonic
	vpubErr     string

	// Camera sender reports (RTCP SR): the camera's own clock for each
	// track. Used for diagnostics: how far our audio and video timelines
	// differ from what the camera says (see maybeLogTimingLocked).
	sr    [2]senderReport // video, audio
	srOff [2]offsetStat
}

type senderReport struct {
	have  bool
	ntp   time.Time // camera clock
	rtp   uint32
	clock uint32
	count int
	first senderReportPoint // first report, for the measured clock rate
}

type senderReportPoint struct {
	ntp time.Time
	rtp uint32
}

// offsetStat accumulates (our timeline - camera clock) for one track.
type offsetStat struct {
	sum time.Duration
	n   int
}

const (
	trackVideo = iota
	trackAAC
	trackG711
)

const (
	maxAlignDelay = time.Second
	maxQueueItems = 600

	// alignment delay decay: after a window in which every track stayed at
	// least decayMargin under the delay, it glides down at decayRate toward
	// the largest lateness seen plus the margin. MediaMTX sends RTSP readers
	// a sender report every 10 s built from packet arrival times, so the
	// delay may change by less than one 20 ms G.711 packet between two
	// reports (1.5 ms/s = 15 ms per report); faster, FFmpeg-based readers
	// would see time step back.
	decayWindow = 10 * time.Second
	decayMargin = 40 * time.Millisecond
	decayRate   = 0.0015

	videoClock = 90000
	// minimum spacing between two published video frames (1 ms)
	minVideoStep = videoClock / 1000
	// the video clock is hard-reset to the wall clock only beyond this
	// distance; closer than that it is steered gently
	videoHardResync = 3 * videoClock
	// a late packet older than this is treated as a camera clock reset
	maxStragglerAge = videoClock
)

type sinkItem struct {
	// capture is the item's place on the shared timeline; it is due at
	// capture + the delay in force when it is delivered, so a delay change
	// applies to queued items too and every track keeps the same offset.
	capture time.Time
	nals    [][]byte // video
	idr     bool
	data    []byte // one AAC access unit, or G.711 samples
	ts      uint32
}

func newNativeSink(rtspURL, label string, wantAudio bool, transcoderPath ...string) *nativeSink {
	s := &nativeSink{rtspURL: rtspURL, label: label, wantAudio: wantAudio, now: time.Now, wake: make(chan struct{}, 1)}
	s.start = s.now()
	if len(transcoderPath) > 0 {
		s.transcoderPath = transcoderPath[0]
	}
	s.asm = newNALAssembler(func(n []byte) { s.auNALs = append(s.auNALs, n) })
	return s
}

// ticksToTime converts ticks of a clock since s.start into wall time
// (split to avoid int64 overflow on long runs).
func (s *nativeSink) ticksToTime(ticks, clock int64) time.Time {
	secs, rem := ticks/clock, ticks%clock
	return s.start.Add(time.Duration(secs)*time.Second + time.Duration(rem*int64(time.Second)/clock))
}

func (s *nativeSink) wallTicks(t time.Time, clock int64) int64 {
	d := t.Sub(s.start)
	secs := int64(d / time.Second)
	rem := int64(d % time.Second)
	return secs*clock + rem*clock/int64(time.Second)
}

// OnVideoPacket consumes one video RTP payload. Access units are delimited by
// the RTP timestamp alone: the camera's marker bits are not relied on (an
// early marker would split a frame into two access units with one timestamp).
func (s *nativeSink) OnVideoPacket(payload []byte, ts uint32, marker bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	now := s.now()
	if s.firstPkt.IsZero() {
		s.firstPkt = now
	}
	// A packet older than the frame being assembled (or than the last
	// finished frame) arrived late: its frame was already handed on, so it is
	// dropped rather than turned into a new frame with an old timestamp. A
	// long run of "old" packets, or a big step back, is a camera clock reset.
	ref, haveRef := s.auTS, s.haveAU
	if !haveRef {
		ref, haveRef = s.lastAUTS, s.haveLastAU
	}
	if haveRef && ts != ref {
		back := -int64(int32(ts - ref))
		if back > 0 && back <= maxStragglerAge && s.oldPktRun < 30 {
			s.oldPktRun++
			s.droppedLate++
			return
		}
	}
	s.oldPktRun = 0
	if s.haveAU && ts != s.auTS {
		s.asm.Flush()
		s.flushAULocked()
	}
	if !s.haveAU {
		s.haveAU = true
		s.auTS = ts
		s.auWall = now
	}
	s.asm.Push(payload)
}

func (s *nativeSink) flushAULocked() {
	nals := s.auNALs
	auTS, auWall := s.auTS, s.auWall
	s.auNALs = nil
	s.haveAU = false
	s.haveLastAU, s.lastAUTS = true, auTS
	if len(nals) == 0 {
		return
	}
	hasIDR, hasSPS := false, false
	for _, n := range nals {
		switch h264NALType(n) {
		case 7:
			if s.sps == nil || string(s.sps) != string(n) {
				s.noteSPSLocked(n)
			}
			s.sps = n
			hasSPS = true
		case 8:
			s.pps = n
		case 5:
			hasIDR = true
		}
	}
	s.countFrameLocked(auWall)
	if s.auHook != nil {
		s.auHook(auTS, nals)
	}
	if !s.ensurePublisherLocked() {
		return
	}
	if hasIDR && !hasSPS && s.sps != nil && s.pps != nil {
		nals = append([][]byte{s.sps, s.pps}, nals...)
	}
	v := s.videoTSLocked(auTS, auWall)
	s.noteSROffsetLocked(0, auTS, s.ticksToTime(v, videoClock))
	if s.alignedLocked() {
		s.writeViewerCopyLocked(nals, hasIDR, uint32(v))
		s.enqueueLocked(trackVideo, sinkItem{nals: nals, idr: hasIDR, ts: uint32(v)}, s.ticksToTime(v, videoClock))
		return
	}
	s.writeVideoLocked(nals, hasIDR, uint32(v))
}

// noteSPSLocked records (and logs once per change) what the camera sends.
func (s *nativeSink) noteSPSLocked(nal []byte) {
	info, err := parseH264SPS(nal)
	if err != nil {
		return
	}
	s.spsInfo, s.haveInfo = info, true
	extra := ""
	if info.FPS > 0 {
		extra = fmt.Sprintf(", %.3g fps declared", info.FPS)
	}
	fmt.Printf("[%s] [publish] video: H.264 %s %s, %dx%d, POC type %d%s\n",
		s.label, info.profileName(), info.levelName(), info.Width, info.Height, info.POCType, extra)
}

// countFrameLocked measures the real frame rate over 10-second windows.
func (s *nativeSink) countFrameLocked(wall time.Time) {
	if s.fpsWin.IsZero() {
		s.fpsWin = wall
		return
	}
	s.fpsCount++
	if el := wall.Sub(s.fpsWin); el >= 10*time.Second {
		s.fps = float64(s.fpsCount) / el.Seconds()
		s.fpsWin, s.fpsCount = wall, 0
	}
}

// VideoInfo reports the stream's size and frame rate once the camera's SPS
// has been seen. The frame rate is measured (the first value appears after
// about 10 seconds); the SPS's own figure is reported separately.
func (s *nativeSink) VideoInfo() (VideoInfo, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.haveInfo {
		return VideoInfo{}, false
	}
	return VideoInfo{
		Width: s.spsInfo.Width, Height: s.spsInfo.Height,
		FPS: roundFPS(s.fps), SPSFPS: roundFPS(s.spsInfo.FPS),
		Profile: s.spsInfo.profileName(), Level: s.spsInfo.levelName(), POCType: s.spsInfo.POCType,
	}, true
}

func roundFPS(f float64) float64 {
	return float64(int64(f*10+0.5)) / 10
}

func (s *nativeSink) writeVideoLocked(nals [][]byte, idr bool, ts uint32) {
	if s.pub == nil {
		return
	}
	if !s.sawIDR {
		if !idr {
			return // decoders cannot start before a keyframe
		}
		s.sawIDR = true
	}
	ts = s.guardLocked(trackVideo, ts)
	if err := s.pub.WriteH264(nals, ts); err != nil {
		fmt.Printf("[%s] [publish] RTSP write failed (%v); reconnecting\n", s.label, err)
		s.setPubLocked(nil)
		return
	}
	s.published++
}

// setPubLocked switches publications; anything queued for the old one is dropped.
func (s *nativeSink) setPubLocked(p *rtspPublisher) {
	if s.pub != nil && s.pub != p {
		s.pub.Close()
	}
	s.pub = p
	s.sawIDR = false
	for i := range s.queues {
		s.queues[i] = nil
		// MediaMTX re-bases each track on a new publication's first packet
		s.guard[i].reset()
		s.guardNote[i] = false
	}
}

// guardLocked passes ts through the track's monotonic guard and logs the
// first hold of each publication.
func (s *nativeSink) guardLocked(track int, ts uint32) uint32 {
	out, held := s.guard[track].next(ts)
	if held > 0 {
		s.guardHeld++
	}
	if held > 0 && !s.guardNote[track] {
		s.guardNote[track] = true
		fmt.Printf("[%s] [publish] %s timestamp stepped back %d ticks; held so the stream stays monotonic\n",
			s.label, [...]string{"video", "AAC", "G.711"}[track], held)
	}
	return out
}

// alignedLocked reports whether deliveries go through the alignment queue.
func (s *nativeSink) alignedLocked() bool {
	return s.pub != nil && s.pub.hasAudio
}

// ensurePublisherLocked returns true when a live publisher is available.
func (s *nativeSink) ensurePublisherLocked() bool {
	if s.pub != nil {
		select {
		case <-s.pub.Done():
			fmt.Printf("[%s] [publish] MediaMTX closed the publication; reconnecting\n", s.label)
			s.setPubLocked(nil)
		default:
			return true
		}
	}
	if s.sps == nil || s.pps == nil {
		return false
	}
	// Give the audio track a moment to appear so it can be announced too.
	if s.wantAudio && s.audioCfg == nil && s.now().Sub(s.firstPkt) < 3*time.Second {
		return false
	}
	if s.now().Sub(s.lastDial) < 2*time.Second {
		return false
	}
	s.lastDial = s.now()
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	var audio *aacConfig
	if s.wantAudio && s.audioCfg != nil {
		audio = s.audioCfg
	}
	withG711 := audio != nil && s.transcoderPath != ""
	pub, err := dialRTSPPublisher(ctx, s.rtspURL, s.sps, s.pps, audio, withG711)
	if err != nil {
		msg := err.Error()
		if msg != s.dialErr {
			fmt.Printf("[%s] [publish] cannot publish to %s: %v (will keep retrying)\n", s.label, s.rtspURL, err)
		}
		s.dialErr = msg
		return false
	}
	s.dialErr = ""
	s.setPubLocked(pub)
	if withG711 && s.g711 == nil {
		s.startG711Locked()
	}
	if audio != nil {
		if s.delay == 0 {
			// starting point: an audio packet is at least its own duration old
			// on arrival, plus transcoding for the browser copy; the delay then
			// adapts to what is actually observed.
			n := s.aFramesPerPkt
			if n < 1 {
				n = 1
			}
			s.delay = time.Duration(n*s.aFrameSamplesLocked()) * time.Second / time.Duration(audio.SampleRate)
			if s.g711 != nil {
				s.delay += 60 * time.Millisecond
			}
			if s.delay < 100*time.Millisecond {
				s.delay = 100 * time.Millisecond
			}
			s.minDelay = s.delay
		}
		if !s.scheduler {
			s.scheduler = true
			go s.runScheduler()
		}
	}
	fmt.Printf("[%s] [publish] publishing to %s (native, audio=%v, browser audio=%v)\n", s.label, s.rtspURL, audio != nil, s.g711 != nil)
	return true
}

// videoTSLocked maps camera RTP time to a strictly increasing 90 kHz
// publication clock (ticks since start). Frame spacing follows the camera's
// clock; the result is gently steered toward wall-clock time so it cannot
// drift away from the audio (which is wall-clock anchored). It never goes
// backwards:
//   - a stall followed by a burst of late frames is carried by the camera
//     clock (the burst's frames keep their real spacing);
//   - a camera clock reset or jump continues one frame after the last frame
//     (or at the wall clock, if that is later);
//   - more than 3 s ahead of the wall clock, frames advance at half speed
//     until it catches up; more than 3 s behind, it jumps forward.
func (s *nativeSink) videoTSLocked(camTS uint32, wall time.Time) int64 {
	const clock = videoClock
	wallNow := s.wallTicks(wall, clock)
	if !s.vAnchored {
		s.vAnchored = true
		s.vBase = wallNow
		s.vElapsed = 0
		s.vLastCam = camTS
		s.vStep = clock / 15
		s.vLastOut = s.vBase
		return s.vBase
	}
	delta := int64(int32(camTS - s.vLastCam))
	s.vLastCam = camTS
	wallElapsed := wallNow - s.vBase
	if delta <= 0 || delta > 5*clock {
		// camera clock reset or jump: the next frame, or the wall clock if later
		s.vElapsed += s.vStep
		if wallElapsed > s.vElapsed {
			s.vElapsed = wallElapsed
		}
	} else {
		if delta <= clock/2 {
			s.vStep = delta
		}
		s.vElapsed += delta
		d := wallElapsed - s.vElapsed
		switch {
		case d > videoHardResync:
			s.vElapsed = wallElapsed
		case d < -videoHardResync:
			s.vElapsed -= delta / 2
		default:
			// slow drift correction; keeps spacing smooth (at most half a frame)
			c := d / 64
			if c < -delta/2 {
				c = -delta / 2
			}
			s.vElapsed += c
		}
	}
	out := s.vBase + s.vElapsed
	if out <= s.vLastOut {
		out = s.vLastOut + minVideoStep
		s.vElapsed = out - s.vBase
	}
	s.vLastOut = out
	s.statMinMax(&s.stVLag, &s.stHaveV, (wallElapsed-s.vElapsed)*1000/clock)
	return out
}

// OnAudioADTS consumes one audio RTP payload (one or more ADTS frames)
// without RTP timing: the timeline follows the wall clock.
func (s *nativeSink) OnAudioADTS(payload []byte) {
	s.onAudio(payload, 0, false)
}

// OnAudioRTP consumes one audio RTP payload with the camera's RTP timestamp,
// which tells lost frames (the timestamp jumps) from late ones (it doesn't).
func (s *nativeSink) OnAudioRTP(payload []byte, rtpTS uint32) {
	s.onAudio(payload, rtpTS, true)
}

// OnAudioRTPClock is OnAudioRTP for a stream whose RTP clock rate is known
// (from the camera's SDP); a clock that differs from the AAC sample rate is
// converted.
func (s *nativeSink) OnAudioRTPClock(payload []byte, rtpTS uint32, clock uint32) {
	if clock > 0 {
		s.mu.Lock()
		s.aClock = clock
		s.mu.Unlock()
	}
	s.onAudio(payload, rtpTS, true)
}

func (s *nativeSink) onAudio(payload []byte, rtpTS uint32, haveRTP bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || !s.wantAudio {
		return
	}
	if s.audioCfg == nil {
		if cfg, ok := aacConfigFromADTS(payload); ok {
			s.audioCfg = &cfg
		} else {
			return
		}
	}
	units := splitADTS(payload)
	if len(units) == 0 {
		return
	}
	if s.aFramesPerPkt == 0 {
		s.aFramesPerPkt = len(units)
	}
	lostFrames, late := s.audioRTPLocked(rtpTS, haveRTP, len(units))
	if late {
		s.droppedAudio++
		return
	}
	if s.pub == nil {
		return
	}
	if !s.pub.hasAudio {
		fmt.Printf("[%s] [publish] audio started after the stream was announced; re-announcing the stream with audio\n", s.label)
		s.setPubLocked(nil)
		return
	}
	rate := int64(s.audioCfg.SampleRate)
	now := s.now()
	s.measureAudioCadenceLocked(rtpTS, haveRTP, now)
	fs := int64(s.aFrameSamplesLocked())
	if s.g711 != nil && s.g711.frameSamples != int(fs) && s.transcoderPath != "" {
		// the frame length was only just learned: restart the browser-audio
		// copy so it is resampled to real time
		s.g711.Close()
		s.g711 = nil
		s.startG711Locked()
	}
	frameDur := time.Duration(fs) * time.Second / time.Duration(rate)
	// the first sample of this payload was captured before its frames were
	// complete: one frame duration per frame
	capture := now.Add(-time.Duration(len(units)) * frameDur)
	expected := s.wallTicks(capture, rate)
	if !s.aAnchored {
		s.aAnchored = true
		s.aNext = expected
	} else {
		s.aNext += int64(lostFrames) * fs
		// Without the camera's RTP clock, lateness and loss look alike, so the
		// wall clock keeps the timeline within a second; with it, a burst of
		// late frames keeps its place and only gross errors are corrected.
		behind := rate
		if s.aRTPStep != 0 {
			behind = 3 * rate
		}
		d := s.aNext - expected // > 0: the timeline is ahead of arrival
		s.noteAudioLeadLocked(d, rate, fs, now)
		switch {
		case d < -behind:
			// far behind the wall clock: audio was lost without the RTP
			// clock saying so; jump forward
			s.aNext = expected
		case d > rate, s.aDropBudget > 0:
			// Ahead of arrival: this audio would be delivered that much
			// late. Skip the frame without advancing the timeline (never
			// step it back); the next frame takes its place.
			if s.aDropBudget > 0 {
				s.aDropBudget--
			}
			s.droppedAudio++
			s.leadDrops++
			return
		}
	}
	first := s.aNext
	s.aNext += fs * int64(len(units))
	if haveRTP {
		s.noteSROffsetLocked(1, rtpTS, s.ticksToTime(first, rate))
	}
	if s.g711 != nil && s.pub.hasG711 {
		s.g711.Feed(payload, len(units), first, s.audioCfg.SampleRate)
	}
	for i, u := range units {
		ts := first + int64(i)*fs
		s.enqueueLocked(trackAAC, sinkItem{data: u, ts: uint32(ts)}, s.ticksToTime(ts, rate))
	}
	s.maybeLogTimingLocked(now)
}

// startG711Locked starts the browser-audio transcoder for the current frame
// length. Output of an earlier transcoder is ignored (generation check).
func (s *nativeSink) startG711Locked() {
	rate := 8000
	if s.audioCfg != nil {
		rate = s.audioCfg.SampleRate
	}
	s.g711Gen++
	gen := s.g711Gen
	t, err := startG711Transcoder(s.transcoderPath, rate, s.aFrameSamplesLocked(), func(b []byte, ts int64) { s.emitG711Gen(gen, b, ts) })
	if err != nil {
		fmt.Printf("[%s] [publish] browser audio track unavailable: %v\n", s.label, err)
		return
	}
	s.g711 = t
}

// aFrameSamplesLocked is the real time one camera AAC frame covers, in
// samples: the measured value, else the camera's RTP step, else 1024.
func (s *nativeSink) aFrameSamplesLocked() int {
	if s.aFrameSamples > 0 {
		return s.aFrameSamples
	}
	if st := s.rtpStepSamplesLocked(); st > 0 {
		return st
	}
	return 1024
}

// rtpStepSamplesLocked converts the learned RTP step to AAC samples.
func (s *nativeSink) rtpStepSamplesLocked() int {
	if s.aRTPStep == 0 || s.audioCfg == nil {
		return 0
	}
	st := int64(s.aRTPStep)
	if s.aClock > 0 && int(s.aClock) != s.audioCfg.SampleRate {
		st = st * int64(s.audioCfg.SampleRate) / int64(s.aClock)
	}
	if st < 480 || st > 2048 {
		return 0
	}
	return int(st)
}

// measureAudioCadenceLocked checks the RTP step against how fast frames
// really arrive, over 20-second windows without gaps. A camera whose RTP
// clock is truthful keeps the step; one that labels 1024-sample frames with
// a smaller step is caught here. Two windows must agree before a value
// already in use changes.
func (s *nativeSink) measureAudioCadenceLocked(ts uint32, haveRTP bool, now time.Time) {
	stepSamples := s.rtpStepSamplesLocked()
	if !haveRTP || stepSamples == 0 {
		return
	}
	defer func() { s.aMeasLast = now }()
	if s.aMeasStart.IsZero() || now.Sub(s.aMeasLast) > time.Second {
		s.aMeasStart, s.aMeasRTP = now, ts
		return
	}
	wall := now.Sub(s.aMeasStart)
	if wall < 20*time.Second {
		return
	}
	rtp := int64(int32(ts - s.aMeasRTP))
	s.aMeasStart, s.aMeasRTP = now, ts
	if rtp <= 0 {
		return
	}
	clock := int64(s.aClock)
	if clock <= 0 {
		clock = int64(s.audioCfg.SampleRate)
	}
	ratio := wall.Seconds() / (float64(rtp) / float64(clock)) // real time per RTP time
	measured := float64(stepSamples) * ratio
	cand := stepSamples
	switch {
	case math.Abs(measured/float64(stepSamples)-1) < 0.03:
		cand = stepSamples
	case math.Abs(measured/1024-1) < 0.03:
		cand = 1024
	default:
		return // no clear answer (clock drift or jitter): keep what we have
	}
	if cand == s.aMeasCand {
		s.aMeasVotes++
	} else {
		s.aMeasCand, s.aMeasVotes = cand, 1
	}
	if (s.aFrameSamples == 0 && s.aMeasVotes >= 1) || s.aMeasVotes >= 2 {
		if s.aFrameSamples != cand {
			fmt.Printf("[%s] [publish] audio: each camera frame covers %d samples (%d ms) of real time\n",
				s.label, cand, int64(cand)*1000/int64(s.audioCfg.SampleRate))
		}
		s.aFrameSamples = cand
	}
}

// noteAudioLeadLocked tracks how far the audio timeline runs ahead of
// arrival and books frame skips when the lead persists for a whole window.
func (s *nativeSink) noteAudioLeadLocked(lead, rate, fs int64, now time.Time) {
	ms := lead * 1000 / rate
	s.statMinMax(&s.stALead, &s.stHaveA, ms)
	if !s.aWinHave {
		s.aWinMinLead, s.aWinMaxLead, s.aWinHave = lead, lead, true
	}
	if lead < s.aWinMinLead {
		s.aWinMinLead = lead
	}
	if lead > s.aWinMaxLead {
		s.aWinMaxLead = lead
	}
	if s.aWinStart.IsZero() {
		s.aWinStart = now
		return
	}
	if now.Sub(s.aWinStart) < 2*time.Second {
		return
	}
	margin := rate * 60 / 1000 // arrival jitter allowance
	switch {
	case s.aWinMinLead > fs+margin && s.aDropBudget == 0:
		// ahead for the whole window: skip that many stale frames
		s.aDropBudget = int((s.aWinMinLead - margin) / fs)
	case s.aWinMaxLead < -(fs + margin):
		// behind for the whole window (not a stall: those catch up within
		// it): move forward, leaving a short gap
		s.aNext += -s.aWinMaxLead - margin
		s.leadGaps++
	}
	s.aWinStart, s.aWinHave = now, false
}

func (s *nativeSink) statMinMax(mm *[2]int64, have *bool, v int64) {
	if !*have {
		mm[0], mm[1], *have = v, v, true
		return
	}
	if v < mm[0] {
		mm[0] = v
	}
	if v > mm[1] {
		mm[1] = v
	}
}

// maybeLogTimingLocked prints a one-line timing summary once a minute, so a
// latency report can be read from the log.
func (s *nativeSink) maybeLogTimingLocked(now time.Time) {
	if s.stAt.IsZero() {
		s.stAt = now
		return
	}
	if now.Sub(s.stAt) < time.Minute {
		return
	}
	line := fmt.Sprintf("[%s] [publish] timing, last minute: A/V delay %d ms", s.label, s.delay.Milliseconds())
	if s.stHaveA {
		line += fmt.Sprintf("; audio ahead of arrival %d..%d ms", s.stALead[0], s.stALead[1])
	}
	if s.stHaveV {
		line += fmt.Sprintf("; video behind arrival %d..%d ms", s.stVLag[0], s.stVLag[1])
	}
	if s.stHG {
		line += fmt.Sprintf("; browser audio ready %d..%d ms after capture", s.stG[0], s.stG[1])
	}
	if s.stGrowAge > 0 {
		line += fmt.Sprintf("; delay last raised by %s arriving %d ms late", trackName(s.stGrowTrack), s.stGrowAge.Milliseconds())
	}
	line += fmt.Sprintf("; audio frames skipped %d, gaps %d; frame length %d", s.leadDrops, s.leadGaps, s.aFrameSamplesLocked())
	if s.lateDropped > 0 {
		line += fmt.Sprintf("; late browser audio dropped %d", s.lateDropped)
	}
	if s.vpub != nil {
		line += "; video-only viewer copy on"
	}
	line += s.srSummaryLocked()
	fmt.Println(line)
	s.stAt = now
	s.stHaveA, s.stHaveV, s.stHG = false, false, false
	s.stGrowAge = 0
}

func trackName(t int) string {
	switch t {
	case trackVideo:
		return "video"
	case trackAAC:
		return "camera audio"
	default:
		return "browser audio"
	}
}

// audioRTPLocked uses the camera's audio RTP clock to tell a late or
// duplicated packet (late=true: drop it) from frames that were lost in
// between (lostFrames > 0: leave a gap on the timeline). The camera's RTP
// ticks per frame are learned from the stream (the WS03 uses 960 per AAC
// frame at "8000 Hz"), so no particular value is assumed.
func (s *nativeSink) audioRTPLocked(ts uint32, haveRTP bool, frames int) (lostFrames int, late bool) {
	if !haveRTP {
		return 0, false
	}
	defer func() {
		if !late {
			s.aHaveRTP, s.aLastRTP, s.aLastFrames = true, ts, frames
		}
	}()
	if !s.aHaveRTP {
		return 0, false
	}
	delta := int64(int32(ts - s.aLastRTP))
	if delta <= 0 {
		if delta > -int64(1<<16) {
			return 0, true // duplicate or reordered
		}
		s.aRTPStep, s.aStepCand, s.aStepVotes = 0, 0, 0 // clock reset: learn again
		return 0, false
	}
	if s.aRTPStep == 0 {
		// learn the step from packets that follow each other directly: three
		// equal per-frame gaps in a row
		if s.aLastFrames > 0 && delta%int64(s.aLastFrames) == 0 {
			cand := uint32(delta / int64(s.aLastFrames))
			if cand == s.aStepCand {
				s.aStepVotes++
			} else {
				s.aStepCand, s.aStepVotes = cand, 1
			}
			if s.aStepVotes >= 3 {
				s.aRTPStep = cand
			}
		}
		return 0, false
	}
	step := int64(s.aRTPStep)
	elapsed := int((delta + step/2) / step) // frames the camera clock advanced
	if lost := elapsed - s.aLastFrames; lost > 0 && lost < 80 {
		return lost, false
	}
	return 0, false
}

// emitG711 is called by the transcoder with µ-law samples on the audio
// timeline (8 kHz ticks since start).
func (s *nativeSink) emitG711(samples []byte, ts int64) {
	s.mu.Lock()
	gen := s.g711Gen
	s.mu.Unlock()
	s.emitG711Gen(gen, samples, ts)
}

func (s *nativeSink) emitG711Gen(gen int, samples []byte, ts int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.pub == nil || !s.pub.hasG711 || gen != s.g711Gen {
		return
	}
	s.statMinMax(&s.stG, &s.stHG, s.now().Sub(s.ticksToTime(ts, 8000)).Milliseconds())
	if ts < s.g711Next {
		return // overlaps what was already sent: keep the track monotonic
	}
	s.g711Next = ts + int64(len(samples))
	s.enqueueLocked(trackG711, sinkItem{data: samples, ts: uint32(ts)}, s.ticksToTime(ts, 8000))
}

// lateSustain is how long a track must keep arriving later than the delay
// before the delay grows. A single late burst (a short network stall, a
// transcoder hiccup) doesn't raise it: counting late packets did, because one
// late 120 ms camera audio frame becomes six 20 ms browser-audio packets.
const lateSustain = time.Second

// lateAudioSlack: browser audio later than the delay by more than this is
// dropped. It can no longer be in step with the picture, and a late burst
// makes the browser hold the picture back too (it keeps sound and picture in
// step even when muted).
const lateAudioSlack = 40 * time.Millisecond

// enqueueLocked schedules an item for delivery at capture+delay. The delay
// grows when a track keeps arriving later than that for lateSustain, and
// shrinks again once every track has been comfortably on time for a while.
func (s *nativeSink) enqueueLocked(track int, it sinkItem, capture time.Time) {
	now := s.now()
	age := now.Sub(capture)
	if age > s.delay {
		if s.lateSince[track].IsZero() {
			s.lateSince[track] = now
		}
		if now.Sub(s.lateSince[track]) >= lateSustain && s.delay < maxAlignDelay && age <= maxAlignDelay {
			nd := age + 20*time.Millisecond
			if nd > maxAlignDelay {
				nd = maxAlignDelay
			}
			if nd > s.delay {
				s.delay = nd
				s.decayTarget = 0
				s.stGrowTrack, s.stGrowAge = track, age
				fmt.Printf("[%s] [publish] audio/video alignment delay now %d ms (%s arrived %d ms after capture for over a second)\n", s.label, s.delay.Milliseconds(), trackName(track), age.Milliseconds())
			}
			s.lateSince[track] = time.Time{}
		}
		if track == trackG711 && age > s.delay+lateAudioSlack {
			s.lateDropped++
			return
		}
	} else {
		s.lateSince[track] = time.Time{}
	}
	if age <= maxAlignDelay {
		s.decayDelayLocked(now, age)
	}
	it.capture = capture
	q := append(s.queues[track], it)
	if len(q) > maxQueueItems {
		q = q[len(q)-maxQueueItems:]
	}
	s.queues[track] = q
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// decayDelayLocked lowers the alignment delay after a window in which every
// item arrived at least decayMargin earlier than the delay required. It
// glides (decayRate) instead of stepping: MediaMTX 1.9 stamps each packet
// with its arrival time for RTSP readers' sender reports, so a faster drop
// in the delay would look like a step back in time to FFmpeg-based readers.
func (s *nativeSink) decayDelayLocked(now time.Time, age time.Duration) {
	if s.winStart.IsZero() {
		s.winStart, s.lastGlide = now, now
	}
	if age > s.winMaxAge {
		s.winMaxAge = age
	}
	// glide toward the target chosen at the end of the last window
	if s.decayTarget > 0 && s.delay > s.decayTarget {
		step := time.Duration(float64(now.Sub(s.lastGlide)) * decayRate)
		if step > 0 {
			nd := s.delay - step
			if nd <= s.decayTarget {
				nd = s.decayTarget
				fmt.Printf("[%s] [publish] audio/video alignment delay now %d ms (tracks on time)\n", s.label, nd.Milliseconds())
			}
			s.delay = nd
		}
	}
	s.lastGlide = now
	if now.Sub(s.winStart) < decayWindow {
		return
	}
	floor := s.minDelay
	if floor <= 0 {
		floor = 100 * time.Millisecond
	}
	s.decayTarget = 0
	if s.delay > floor && s.winMaxAge+decayMargin < s.delay {
		target := s.winMaxAge + decayMargin
		if target < floor {
			target = floor
		}
		s.decayTarget = target
	}
	s.winStart, s.winMaxAge = now, 0
}

// runScheduler hands queued items to the publisher when they are due.
func (s *nativeSink) runScheduler() {
	for {
		s.mu.Lock()
		if s.closed {
			s.scheduler = false
			s.mu.Unlock()
			return
		}
		next := -1
		for i := range s.queues {
			if len(s.queues[i]) > 0 && (next < 0 || s.queues[i][0].capture.Before(s.queues[next][0].capture)) {
				next = i
			}
		}
		wait := time.Second
		if next >= 0 {
			it := s.queues[next][0]
			if w := it.capture.Add(s.delay).Sub(s.now()); w > 0 {
				wait = w
			} else {
				s.queues[next] = s.queues[next][1:]
				s.deliverLocked(next, it)
				s.mu.Unlock()
				continue
			}
		}
		s.mu.Unlock()
		t := time.NewTimer(wait)
		select {
		case <-t.C:
		case <-s.wake:
		}
		t.Stop()
	}
}

func (s *nativeSink) deliverLocked(track int, it sinkItem) {
	if s.pub == nil {
		return
	}
	switch track {
	case trackVideo:
		s.writeVideoLocked(it.nals, it.idr, it.ts)
	case trackAAC:
		if err := s.pub.WriteAAC([][]byte{it.data}, s.guardLocked(trackAAC, it.ts)); err != nil {
			s.setPubLocked(nil)
		}
	case trackG711:
		_ = s.pub.WriteG711(it.data, s.guardLocked(trackG711, it.ts))
	}
}

// Close ends the publication.
func (s *nativeSink) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	if s.g711 != nil {
		s.g711.Close()
		s.g711 = nil
	}
	if s.vpub != nil {
		s.vpub.Close()
		s.vpub = nil
	}
	s.setPubLocked(nil)
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// ---- viewer copy (video only, no alignment delay) ----

// ViewerPathSuffix is appended to a camera's publication path for the
// video-only viewer copy.
const ViewerPathSuffix = "/viewer"

// writeViewerCopyLocked sends one access unit to the viewer copy at once,
// connecting it first if needed (in the background, so the camera's media
// is never held up by a dial).
func (s *nativeSink) writeViewerCopyLocked(nals [][]byte, idr bool, ts uint32) {
	if s.vpub != nil {
		select {
		case <-s.vpub.Done():
			s.vpub = nil
		default:
		}
	}
	if s.vpub == nil {
		s.dialViewerCopyLocked()
		return
	}
	if !s.vpubSawIDR {
		if !idr {
			return
		}
		s.vpubSawIDR = true
	}
	out, _ := s.vpubGuard.next(ts)
	if err := s.vpub.WriteH264(nals, out); err != nil {
		s.vpub.Close()
		s.vpub = nil
	}
}

func (s *nativeSink) dialViewerCopyLocked() {
	if s.vpubDialing || s.closed || s.sps == nil || s.pps == nil || s.now().Sub(s.vpubLast) < 5*time.Second {
		return
	}
	s.vpubDialing, s.vpubLast = true, s.now()
	url, sps, pps := s.rtspURL+ViewerPathSuffix, s.sps, s.pps
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		pub, err := dialRTSPPublisher(ctx, url, sps, pps, nil)
		s.mu.Lock()
		defer s.mu.Unlock()
		s.vpubDialing = false
		if err != nil {
			if msg := err.Error(); msg != s.vpubErr {
				s.vpubErr = msg
				fmt.Printf("[%s] [publish] video-only viewer copy unavailable: %v\n", s.label, err)
			}
			return
		}
		if s.closed {
			pub.Close()
			return
		}
		s.vpubErr = ""
		s.vpub, s.vpubSawIDR = pub, false
		s.vpubGuard.reset()
		fmt.Printf("[%s] [publish] video-only viewer copy at %s (no alignment delay)\n", s.label, url)
	}()
}

// ViewerCopyReady reports whether the video-only viewer copy is live.
func (s *nativeSink) ViewerCopyReady() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.vpub != nil && s.vpubSawIDR
}

// ---- camera sender reports (diagnostics) ----

// ntpToTime converts a 64-bit NTP timestamp.
func ntpToTime(ntp uint64) time.Time {
	const ntpEpochOffset = 2208988800
	secs := int64(ntp>>32) - ntpEpochOffset
	frac := int64(ntp & 0xFFFFFFFF)
	return time.Unix(secs, frac*int64(time.Second)>>32)
}

// OnSenderReport records an RTCP sender report from the camera for its video
// (track 0) or audio (track 1).
func (s *nativeSink) OnSenderReport(track int, ntp uint64, rtpTS uint32, clock uint32) {
	if track < 0 || track > 1 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r := &s.sr[track]
	t := ntpToTime(ntp)
	if !r.have {
		r.first = senderReportPoint{ntp: t, rtp: rtpTS}
		fmt.Printf("[%s] [publish] the camera sends sender reports for its %s (its own clock)\n", s.label, [...]string{"video", "audio"}[track])
	}
	r.have, r.ntp, r.rtp, r.clock = true, t, rtpTS, clock
	r.count++
}

// noteSROffsetLocked compares our timeline's time for a packet with the
// camera's clock (from its latest sender report).
func (s *nativeSink) noteSROffsetLocked(track int, rtpTS uint32, ours time.Time) {
	r := &s.sr[track]
	if !r.have || r.clock == 0 {
		return
	}
	d := int64(int32(rtpTS - r.rtp))
	cam := r.ntp.Add(time.Duration(d * int64(time.Second) / int64(r.clock)))
	st := &s.srOff[track]
	st.sum += ours.Sub(cam)
	st.n++
}

// srSummaryLocked describes, for the timing line, how our audio and video
// timelines line up according to the camera's own clock.
func (s *nativeSink) srSummaryLocked() string {
	if !s.sr[0].have && !s.sr[1].have {
		return "; no camera sender reports"
	}
	out := ""
	if s.srOff[0].n > 0 && s.srOff[1].n > 0 {
		v := s.srOff[0].sum / time.Duration(s.srOff[0].n)
		a := s.srOff[1].sum / time.Duration(s.srOff[1].n)
		out += fmt.Sprintf("; by the camera's clock our audio is %d ms later than our video", (a - v).Milliseconds())
	}
	if r := s.sr[1]; r.have && r.count > 1 {
		el := r.ntp.Sub(r.first.ntp).Seconds()
		if el > 5 {
			out += fmt.Sprintf("; camera audio clock %.0f Hz", float64(int32(r.rtp-r.first.rtp))/el)
		}
	}
	s.srOff = [2]offsetStat{}
	return out
}

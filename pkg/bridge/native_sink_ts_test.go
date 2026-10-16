package bridge

import (
	"encoding/hex"
	"math/rand"
	"net"
	"strings"
	"testing"
	"time"
)

// fakeClock drives a nativeSink's notion of "now" in tests.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newTestSink(t *testing.T, audio bool) (*nativeSink, *fakeClock) {
	t.Helper()
	clk := &fakeClock{t: time.Date(2026, 9, 28, 12, 0, 0, 0, time.UTC)}
	s := newNativeSink("rtsp://127.0.0.1:1/unused", "test", audio)
	s.now = clk.now
	s.start = clk.t
	return s, clk
}

func mustHex(t *testing.T, h string) []byte {
	t.Helper()
	b, err := hex.DecodeString(h)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestParseH264SPS(t *testing.T) {
	// SPSs produced by x264 (ffmpeg testsrc) at the stated size and rate.
	cases := []struct {
		name, sps      string
		w, h, poc      int
		profile, level string
		fps            float64
	}{
		{"640x360 baseline", "6742c016d900a02ff970110000030001000003001e0f162e48", 640, 360, 2, "Baseline", "2.2", 15},
		{"1080p high with B-frames", "67640028acd940780227e5c044000003000400000300783c60c658", 1920, 1080, 0, "High", "4.0", 15},
		{"720p main", "674d401fd9005005bb011000000300100000030320f1832480", 1280, 720, 2, "Main", "3.1", 25},
		{"1440p high", "67640032acb20050016b6022000003000200000300281e306490", 2560, 1440, 2, "High", "5.0", 10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, err := parseH264SPS(mustHex(t, tc.sps))
			if err != nil {
				t.Fatal(err)
			}
			if s.Width != tc.w || s.Height != tc.h {
				t.Errorf("size %dx%d, want %dx%d", s.Width, s.Height, tc.w, tc.h)
			}
			if s.POCType != tc.poc {
				t.Errorf("POC type %d, want %d", s.POCType, tc.poc)
			}
			if s.profileName() != tc.profile || s.levelName() != tc.level {
				t.Errorf("profile %s level %s, want %s %s", s.profileName(), s.levelName(), tc.profile, tc.level)
			}
			if s.FPS != tc.fps {
				t.Errorf("fps %v, want %v", s.FPS, tc.fps)
			}
		})
	}
	// cut off inside the VUI: the size is still known
	full := mustHex(t, cases[0].sps)
	if s, err := parseH264SPS(full[:12]); err != nil || s.Width != 640 {
		t.Errorf("truncated VUI: %+v %v", s, err)
	}
	for _, bad := range [][]byte{nil, {0x67}, {0x68, 1, 2, 3, 4}, full[:5]} {
		if _, err := parseH264SPS(bad); err == nil {
			t.Errorf("parseH264SPS(%x) should fail", bad)
		}
	}
}

// videoRun feeds frames to videoTSLocked and checks that every output is
// strictly later than the one before.
type videoRun struct {
	t    *testing.T
	s    *nativeSink
	clk  *fakeClock
	cam  uint32
	last int64
	n    int
}

func (r *videoRun) frame(camStep uint32, wallStep time.Duration) int64 {
	r.t.Helper()
	r.cam += camStep
	r.clk.advance(wallStep)
	out := r.s.videoTSLocked(r.cam, r.clk.now())
	if r.n > 0 && out <= r.last {
		r.t.Fatalf("frame %d: timestamp %d is not after %d", r.n, out, r.last)
	}
	r.last, r.n = out, r.n+1
	return out
}

// behindWall is how far the published clock is behind wall time (seconds).
func (r *videoRun) behindWall() float64 {
	return float64(r.s.wallTicks(r.clk.now(), videoClock)-r.last) / videoClock
}

func TestVideoTimestamps_SteadyStream(t *testing.T) {
	s, clk := newTestSink(t, false)
	r := &videoRun{t: t, s: s, clk: clk, cam: 1000}
	var prev int64
	for i := 0; i < 300; i++ {
		out := r.frame(6000, time.Second/15)
		if i > 0 && (out-prev < 5900 || out-prev > 6100) {
			t.Fatalf("frame %d spacing %d, want ~6000", i, out-prev)
		}
		prev = out
	}
}

// A stall of more than a second followed by a burst of the delayed frames
// must not step the timestamps back ("DTS is not monotonically increasing"
// in MediaMTX's HLS muxer).
func TestVideoTimestamps_StallThenBurstNeverGoesBack(t *testing.T) {
	for _, fps := range []int{15, 8} {
		s, clk := newTestSink(t, false)
		r := &videoRun{t: t, s: s, clk: clk}
		step := uint32(videoClock / fps)
		interval := time.Second / time.Duration(fps)
		for round := 0; round < 5; round++ {
			for i := 0; i < 20*fps; i++ {
				r.frame(step, interval)
			}
			// 1.5 s of silence, then the frames captured meanwhile arrive at once
			clk.advance(1500 * time.Millisecond)
			burst := int(1.5 * float64(fps))
			for i := 0; i < burst; i++ {
				r.frame(step, 2*time.Millisecond)
			}
		}
		// back in step with the wall clock after a while
		for i := 0; i < 60*fps; i++ {
			r.frame(step, interval)
		}
		if b := r.behindWall(); b < -0.1 || b > 0.1 {
			t.Errorf("fps %d: %.3f s off the wall clock after recovery", fps, b)
		}
	}
}

func TestVideoTimestamps_CameraClockResetsAndJumps(t *testing.T) {
	s, clk := newTestSink(t, false)
	r := &videoRun{t: t, s: s, clk: clk, cam: 4000000000} // near the uint32 wrap
	for i := 0; i < 100; i++ {
		r.frame(6000, time.Second/15) // wraps past 2^32
	}
	r.cam = 10 // camera restarted its clock
	r.frame(0, time.Second/15)
	for i := 0; i < 50; i++ {
		r.frame(6000, time.Second/15)
	}
	r.frame(20*videoClock, time.Second/15) // 20 s jump forward
	for i := 0; i < 50; i++ {
		r.frame(6000, time.Second/15)
	}
	r.frame(^uint32(0)-3000, time.Second/15) // small step back
	for i := 0; i < 50; i++ {
		r.frame(6000, time.Second/15)
	}
	if b := r.behindWall(); b < -0.5 || b > 0.5 {
		t.Errorf("%.3f s off the wall clock", b)
	}
}

func TestVideoTimestamps_CameraClockDrift(t *testing.T) {
	for _, camStep := range []uint32{5880, 6120} { // camera clock 2% slow / fast
		s, clk := newTestSink(t, false)
		r := &videoRun{t: t, s: s, clk: clk}
		for i := 0; i < 15*600; i++ { // ten minutes
			r.frame(camStep, time.Second/15)
		}
		if b := r.behindWall(); b < -0.2 || b > 0.2 {
			t.Errorf("step %d: %.3f s off the wall clock after 10 minutes", camStep, b)
		}
	}
}

func TestVideoTimestamps_FarAheadSlowsDown(t *testing.T) {
	s, clk := newTestSink(t, false)
	r := &videoRun{t: t, s: s, clk: clk}
	for i := 0; i < 30; i++ {
		r.frame(6000, time.Second/15)
	}
	// a camera that sends 10 s of frames in half a second
	for i := 0; i < 150; i++ {
		r.frame(6000, 3*time.Millisecond)
	}
	for i := 0; i < 15*40; i++ {
		r.frame(6000, time.Second/15)
	}
	if b := r.behindWall(); b < -0.3 || b > 0.3 {
		t.Errorf("%.3f s off the wall clock", b)
	}
}

func TestOnVideoPacket_DropsLatePacketsOfFinishedFrames(t *testing.T) {
	s, clk := newTestSink(t, false)
	var got []uint32
	s.auHook = func(ts uint32, nals [][]byte) { got = append(got, ts) }
	slice := func() []byte { return []byte{0x41, 0x9a, 1, 2, 3} }
	send := func(ts uint32) {
		clk.advance(10 * time.Millisecond)
		s.OnVideoPacket(slice(), ts, true)
	}
	send(1000)
	send(7000)  // flushes 1000
	send(1000)  // straggler from the finished frame: dropped
	send(7000)  // same frame continues
	send(13000) // flushes 7000
	send(19000) // flushes 13000
	want := []uint32{1000, 7000, 13000}
	if len(got) != len(want) {
		t.Fatalf("frames %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frames %v, want %v", got, want)
		}
	}
	if s.droppedLate != 1 {
		t.Errorf("dropped %d late packets, want 1", s.droppedLate)
	}
	// a camera clock reset (large step back) is accepted at once
	var reset uint32 = 19000
	reset -= 5 * videoClock // wraps below zero
	send(reset)
	send(reset + 6000)
	if got[len(got)-1] != reset {
		t.Errorf("clock reset not accepted: %v", got)
	}
	// a small step back that persists is a reset too, after a short run
	small := reset + 6000 - 30000
	for i := 0; i < 40; i++ {
		send(small)
	}
	send(small + 6000)
	if got[len(got)-1] != small {
		t.Errorf("persistent small step back not accepted: %v", got)
	}
}

// audioRun feeds AAC packets (one ADTS frame each) with RTP timestamps.
type audioRun struct {
	t   *testing.T
	s   *nativeSink
	clk *fakeClock
	rtp uint32
}

var testADTS = func() []byte {
	// one ADTS frame: AAC-LC, 8 kHz (index 11), mono, 20 bytes total
	f := make([]byte, 20)
	f[0], f[1] = 0xFF, 0xF1
	f[2] = 1<<6 | 11<<2 // profile LC(1), freq index 11
	f[3] = 1 << 6       // channel config 1
	l := len(f)
	f[3] |= byte(l >> 11 & 0x03)
	f[4] = byte(l >> 3)
	f[5] = byte(l&7)<<5 | 0x1F
	f[6] = 0xFC
	return f
}()

func TestLateAAC_ReplacesVideoOnlyPublication(t *testing.T) {
	s, clk := newTestSink(t, true)
	firstClient, firstServer := net.Pipe()
	defer firstServer.Close()
	first := &rtspPublisher{conn: firstClient, done: make(chan struct{})}
	s.pub = first
	s.lastDial = clk.now()

	s.OnAudioRTP(testADTS, 5000)

	if s.audioCfg == nil {
		t.Fatal("AAC configuration was not learned")
	}
	if s.pub != nil {
		t.Fatal("video-only publication was not removed")
	}
	if !s.lastDial.Equal(clk.now()) {
		t.Fatal("late-audio transition bypassed the existing reconnect backoff")
	}
	select {
	case <-first.Done():
	default:
		t.Fatal("first video-only publication was not closed")
	}

	// A later video-only publication in the same sink must be recoverable too;
	// logging state must never suppress the functional transition.
	secondClient, secondServer := net.Pipe()
	defer secondServer.Close()
	second := &rtspPublisher{conn: secondClient, done: make(chan struct{})}
	s.pub = second
	s.OnAudioRTP(testADTS, 5960)
	if s.pub != nil {
		t.Fatal("second video-only publication was not removed")
	}
	select {
	case <-second.Done():
	default:
		t.Fatal("second video-only publication was not closed")
	}
}

func TestLateAAC_InvalidADTSKeepsVideoOnlyPublication(t *testing.T) {
	s, _ := newTestSink(t, true)
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	old := &rtspPublisher{conn: client, done: make(chan struct{})}
	s.pub = old

	s.OnAudioRTP([]byte{0x01, 0x02, 0x03}, 5000)

	if s.audioCfg != nil {
		t.Fatal("invalid ADTS produced an AAC configuration")
	}
	if s.pub != old {
		t.Fatal("invalid ADTS replaced the video-only publication")
	}
	select {
	case <-old.Done():
		t.Fatal("invalid ADTS closed the video-only publication")
	default:
	}
}

func (r *audioRun) packet(rtpStep uint32, wallStep time.Duration) {
	r.rtp += rtpStep
	r.clk.advance(wallStep)
	r.s.OnAudioRTP(testADTS, r.rtp)
}

func (r *audioRun) timestamps() []uint32 {
	var out []uint32
	for _, it := range r.s.queues[trackAAC] {
		out = append(out, it.ts)
	}
	return out
}

func newAudioRun(t *testing.T) *audioRun {
	s, clk := newTestSink(t, true)
	s.pub = &rtspPublisher{hasAudio: true, done: make(chan struct{})}
	s.delay, s.minDelay = 100*time.Millisecond, 100*time.Millisecond
	return &audioRun{t: t, s: s, clk: clk, rtp: 5000}
}

func checkIncreasing(t *testing.T, ts []uint32) {
	t.Helper()
	for i := 1; i < len(ts); i++ {
		if int32(ts[i]-ts[i-1]) <= 0 {
			t.Fatalf("audio timestamp %d (%d) is not after %d", i, ts[i], ts[i-1])
		}
	}
}

func TestAudioTimestamps_StallThenBurstKeepsEveryFrame(t *testing.T) {
	r := newAudioRun(t)
	frame := 120 * time.Millisecond // the WS03: one frame every 120 ms, RTP +960
	for i := 0; i < 50; i++ {
		r.packet(960, frame)
	}
	r.clk.advance(1500 * time.Millisecond)
	for i := 0; i < 12; i++ { // the frames captured during the stall, at once
		r.packet(960, 2*time.Millisecond)
	}
	for i := 0; i < 50; i++ {
		r.packet(960, frame)
	}
	ts := r.timestamps()
	checkIncreasing(t, ts)
	if len(ts) != 112 || r.s.droppedAudio != 0 {
		t.Fatalf("%d frames queued, %d dropped; want 112 and 0", len(ts), r.s.droppedAudio)
	}
	for i := 1; i < len(ts); i++ {
		want := uint32(960) // once the camera's RTP step is known (4th frame)
		if i < 4 {
			want = 1024
		}
		if ts[i]-ts[i-1] != want {
			t.Fatalf("frame %d: gap %d, want %d (contiguous audio, real time per frame)", i, ts[i]-ts[i-1], want)
		}
	}
}

func TestAudioTimestamps_LostFramesLeaveAGap(t *testing.T) {
	r := newAudioRun(t)
	for i := 0; i < 20; i++ {
		r.packet(960, 120*time.Millisecond)
	}
	r.packet(960*6, 6*120*time.Millisecond) // 5 frames lost on the way
	for i := 0; i < 5; i++ {
		r.packet(960, 120*time.Millisecond)
	}
	ts := r.timestamps()
	checkIncreasing(t, ts)
	if gap := ts[20] - ts[19]; gap != 6*960 {
		t.Errorf("gap after the loss is %d samples, want %d", gap, 6*960)
	}
}

// audioLead is how far the audio timeline is ahead of arrival right now.
func audioLead(r *audioRun) time.Duration {
	exp := r.s.wallTicks(r.clk.now(), 8000)
	return time.Duration(r.s.aNext-exp) * time.Second / 8000
}

// The WS03 sends an audio frame every 120 ms; booking 1024 samples per frame
// would run ahead of real time by 67 ms every second. The timeline must
// follow real time instead.
func TestAudioTimeline_WS03CadenceStaysOnTime(t *testing.T) {
	r := newAudioRun(t)
	rng := rand.New(rand.NewSource(3))
	var worst time.Duration
	for i := 0; i < 5000; i++ { // 10 minutes
		jitter := time.Duration(rng.Intn(30)-15) * time.Millisecond
		r.packet(960, 120*time.Millisecond+jitter)
		r.clk.advance(-jitter)
		r.s.queues[trackAAC] = nil
		if l := audioLead(r); i > 50 && l > worst {
			worst = l
		}
	}
	if worst > 200*time.Millisecond {
		t.Fatalf("audio ran up to %v ahead of real time", worst)
	}
	if r.s.aFrameSamples != 960 {
		t.Errorf("frame length %d, want 960", r.s.aFrameSamples)
	}
	if r.s.leadDrops > 2 {
		t.Errorf("%d frames skipped on a steady stream", r.s.leadDrops)
	}
}

// A camera that really sends 1024-sample frames (one every 128 ms) but steps
// its RTP clock by 960 is recognised from the arrival cadence.
func TestAudioTimeline_MislabelledStepIsCaught(t *testing.T) {
	r := newAudioRun(t)
	for i := 0; i < 2000; i++ { // about 4 minutes
		r.packet(960, 128*time.Millisecond)
		r.s.queues[trackAAC] = nil
	}
	if r.s.aFrameSamples != 1024 {
		t.Fatalf("frame length %d, want 1024 from the cadence", r.s.aFrameSamples)
	}
	if l := audioLead(r); l > 200*time.Millisecond || l < -200*time.Millisecond {
		t.Fatalf("audio %v off real time", l)
	}
}

// At session start the camera may hand over a backlog of audio at once. The
// timeline is anchored on the first frame, so the backlog would put the
// audio seconds ahead for the whole session; stale frames are skipped.
func TestAudioTimeline_StartupBacklogIsSkipped(t *testing.T) {
	r := newAudioRun(t)
	for i := 0; i < 20; i++ { // 2.4 s of audio within 0.4 s
		r.packet(960, 20*time.Millisecond)
	}
	for i := 0; i < 100; i++ { // then 12 s in real time
		r.packet(960, 120*time.Millisecond)
	}
	if l := audioLead(r); l > 200*time.Millisecond {
		t.Fatalf("audio still %v ahead of real time after 12 s", l)
	}
	checkIncreasing(t, r.timestamps())
	if r.s.leadDrops == 0 {
		t.Error("no stale frames were skipped")
	}
}

func TestAudioTimestamps_DuplicatesAndReorderingDropped(t *testing.T) {
	r := newAudioRun(t)
	for i := 0; i < 10; i++ {
		r.packet(960, 120*time.Millisecond)
	}
	r.packet(0, time.Millisecond)              // duplicate
	r.packet(^uint32(0)-959, time.Millisecond) // one frame older
	r.packet(960*2, 120*time.Millisecond)
	checkIncreasing(t, r.timestamps())
	if r.s.droppedAudio != 2 {
		t.Errorf("dropped %d, want 2", r.s.droppedAudio)
	}
}

func TestAudioTimestamps_WithoutRTPNeverGoesBack(t *testing.T) {
	s, clk := newTestSink(t, true)
	s.pub = &rtspPublisher{hasAudio: true, done: make(chan struct{})}
	s.delay = 100 * time.Millisecond
	for round := 0; round < 4; round++ {
		for i := 0; i < 40; i++ {
			clk.advance(128 * time.Millisecond)
			s.OnAudioADTS(testADTS)
		}
		clk.advance(2 * time.Second)
		for i := 0; i < 16; i++ {
			clk.advance(time.Millisecond)
			s.OnAudioADTS(testADTS)
		}
	}
	var ts []uint32
	for _, it := range s.queues[trackAAC] {
		ts = append(ts, it.ts)
	}
	checkIncreasing(t, ts)
}

func TestG711Timestamps_OverlapDropped(t *testing.T) {
	s, _ := newTestSink(t, true)
	s.pub = &rtspPublisher{hasAudio: true, hasG711: true, done: make(chan struct{})}
	s.emitG711(make([]byte, 160), 1000)
	s.emitG711(make([]byte, 160), 1100) // overlaps the first chunk
	s.emitG711(make([]byte, 160), 1160)
	if n := len(s.queues[trackG711]); n != 2 {
		t.Fatalf("%d G.711 chunks queued, want 2", n)
	}
}

// The alignment delay grows while audio arrives late and shrinks back once
// everything has been on time for a while.
func TestAlignmentDelay_RisesThenDecays(t *testing.T) {
	s, clk := newTestSink(t, true)
	s.delay, s.minDelay = 160*time.Millisecond, 160*time.Millisecond
	enqueue := func(track int, age time.Duration) {
		s.enqueueLocked(track, sinkItem{}, clk.now().Add(-age))
		s.queues[track] = nil
	}
	for i := 0; i < 50; i++ { // late audio
		clk.advance(128 * time.Millisecond)
		enqueue(trackAAC, 450*time.Millisecond)
		enqueue(trackVideo, 30*time.Millisecond)
	}
	if s.delay < 450*time.Millisecond {
		t.Fatalf("delay %v did not rise with late audio", s.delay)
	}
	peak := s.delay
	for i := 0; i < 20*8; i++ { // 20 s on time
		clk.advance(128 * time.Millisecond)
		enqueue(trackAAC, 60*time.Millisecond)
		enqueue(trackVideo, 30*time.Millisecond)
	}
	if s.delay >= peak {
		t.Fatalf("delay %v did not fall from %v", s.delay, peak)
	}
	// it glides: never more than 15 ms per 10 s (one MediaMTX sender report)
	prev, prevAt := s.delay, clk.now()
	for i := 0; i < 400*8; i++ { // about seven more minutes
		clk.advance(128 * time.Millisecond)
		enqueue(trackAAC, 60*time.Millisecond)
		enqueue(trackVideo, 30*time.Millisecond)
		if clk.now().Sub(prevAt) >= 10*time.Second {
			if drop := prev - s.delay; drop > 16*time.Millisecond {
				t.Fatalf("delay fell %v in 10 s", drop)
			}
			prev, prevAt = s.delay, clk.now()
		}
	}
	if s.delay != s.minDelay {
		t.Fatalf("delay %v, want back at its starting point %v", s.delay, s.minDelay)
	}
	// a short late burst does not undo the decay; lateness that lasts over
	// a second raises it again
	for i := 0; i < 6; i++ {
		enqueue(trackAAC, 300*time.Millisecond)
	}
	if s.delay != s.minDelay {
		t.Fatalf("a late burst changed the delay to %v", s.delay)
	}
	for i := 0; i < 10; i++ {
		clk.advance(128 * time.Millisecond)
		enqueue(trackAAC, 300*time.Millisecond)
	}
	if s.delay < 300*time.Millisecond {
		t.Fatalf("delay %v did not rise again", s.delay)
	}
}

// One late 120 ms camera frame (six 20 ms browser-audio packets) does not
// raise the shared delay; the late packets, which can't be in step any more,
// are dropped.
func TestAlignmentDelay_SingleLateBurstIgnored(t *testing.T) {
	s, clk := newTestSink(t, true)
	s.delay, s.minDelay = 180*time.Millisecond, 180*time.Millisecond
	s.enqueueLocked(trackVideo, sinkItem{}, clk.now())
	for i := 0; i < 6; i++ {
		s.enqueueLocked(trackG711, sinkItem{}, clk.now().Add(-500*time.Millisecond+time.Duration(i)*20*time.Millisecond))
	}
	if s.delay != 180*time.Millisecond {
		t.Fatalf("delay %v after one late burst", s.delay)
	}
	if n := len(s.queues[trackG711]); n != 0 || s.lateDropped != 6 {
		t.Fatalf("%d late browser-audio packets queued, %d dropped", n, s.lateDropped)
	}
	// a late camera-audio (AAC) frame is still passed on: recorders place it
	// by its timestamp
	s.enqueueLocked(trackAAC, sinkItem{}, clk.now().Add(-500*time.Millisecond))
	if len(s.queues[trackAAC]) != 1 || s.delay != 180*time.Millisecond {
		t.Fatalf("late AAC: queued %d, delay %v", len(s.queues[trackAAC]), s.delay)
	}
}

// Browser audio that stays late (a slow transcoder) raises the delay after
// a second, and is then kept.
func TestAlignmentDelay_SustainedLatenessRaises(t *testing.T) {
	s, clk := newTestSink(t, true)
	s.delay, s.minDelay = 180*time.Millisecond, 180*time.Millisecond
	for i := 0; i < 80; i++ { // 1.6 s of 20 ms packets, each 300 ms old
		clk.advance(20 * time.Millisecond)
		s.enqueueLocked(trackG711, sinkItem{}, clk.now().Add(-300*time.Millisecond))
	}
	if s.delay < 300*time.Millisecond || s.delay > 340*time.Millisecond {
		t.Fatalf("delay %v, want about 320 ms", s.delay)
	}
	if len(s.queues[trackG711]) < 25 {
		t.Fatalf("only %d packets kept after the delay rose", len(s.queues[trackG711]))
	}
}

// Stale browser audio (FFmpeg's leftover after a pause such as Talk) is
// dropped instead of raising the delay to 1 s and bursting out.
func TestAlignmentDelay_StaleBrowserAudioDropped(t *testing.T) {
	s, clk := newTestSink(t, true)
	s.pub = &rtspPublisher{hasAudio: true, hasG711: true, done: make(chan struct{})}
	s.delay, s.minDelay = 180*time.Millisecond, 180*time.Millisecond
	clk.advance(30 * time.Second)
	for i := 0; i < 100; i++ { // 2 s of 27-second-old packets
		clk.advance(20 * time.Millisecond)
		s.emitG711(make([]byte, 160), 3*8000+int64(i*160))
	}
	if len(s.queues[trackG711]) != 0 || s.delay != 180*time.Millisecond {
		t.Fatalf("stale audio: queued %d, delay %v", len(s.queues[trackG711]), s.delay)
	}
}

// Queued items are due at capture + the delay in force when they go out,
// so a delay change moves every track together.
func TestAlignmentDelay_QueuedItemsFollowDelay(t *testing.T) {
	s, clk := newTestSink(t, true)
	s.delay, s.minDelay = 180*time.Millisecond, 180*time.Millisecond
	s.enqueueLocked(trackVideo, sinkItem{}, clk.now())
	for i := 0; i < 70; i++ {
		clk.advance(20 * time.Millisecond)
		s.enqueueLocked(trackG711, sinkItem{}, clk.now().Add(-400*time.Millisecond))
	}
	if s.delay <= 400*time.Millisecond {
		t.Fatalf("delay %v did not rise", s.delay)
	}
	v := s.queues[trackVideo][0]
	g := s.queues[trackG711][len(s.queues[trackG711])-1]
	// the video captured first is still due before later browser audio
	if !v.capture.Add(s.delay).Before(g.capture.Add(s.delay)) {
		t.Fatal("queued video would go out after later audio")
	}
}

func TestVideoInfo_FromSPSAndMeasuredRate(t *testing.T) {
	s, clk := newTestSink(t, false)
	sps := mustHex(t, "67640028acd940780227e5c044000003000400000300783c60c658")
	cam := uint32(0)
	for i := 0; i < 15*25; i++ {
		clk.advance(time.Second / 12) // the camera really sends 12 fps
		cam += 7500
		if i%30 == 0 {
			s.OnVideoPacket(sps, cam, false)
		}
		s.OnVideoPacket([]byte{0x41, 0x9a, 1, 2}, cam, true)
	}
	info, ok := s.VideoInfo()
	if !ok || info.Width != 1920 || info.Height != 1080 || info.SPSFPS != 15 {
		t.Fatalf("info %+v ok=%v", info, ok)
	}
	if info.FPS < 11.5 || info.FPS > 12.5 {
		t.Fatalf("measured %v fps, want ~12", info.FPS)
	}
}

// The camera's sender reports give its own clock for each track; the timing
// line reports how our audio and video timelines compare with it.
func TestSenderReports_OffsetSummary(t *testing.T) {
	s, clk := newTestSink(t, true)
	camStart := time.Date(2026, 9, 29, 10, 0, 0, 0, time.UTC)
	toNTP := func(tm time.Time) uint64 {
		secs := uint64(tm.Unix() + 2208988800)
		frac := uint64(tm.Nanosecond()) << 32 / uint64(time.Second)
		return secs<<32 | frac
	}
	if got := ntpToTime(toNTP(camStart)); got.Sub(camStart).Abs() > time.Microsecond {
		t.Fatalf("ntp round trip %v", got)
	}
	s.OnSenderReport(0, toNTP(camStart), 90000, 90000)
	s.OnSenderReport(1, toNTP(camStart), 8000, 8000)
	// our timeline: video at camera time + 1 s, audio at camera time + 1.25 s
	ours := clk.now()
	s.noteSROffsetLocked(0, 90000+9000, ours.Add(100*time.Millisecond))
	s.noteSROffsetLocked(1, 8000+800, ours.Add(350*time.Millisecond))
	sum := s.srSummaryLocked()
	// offsets: video = ours+100ms - (cam+100ms); audio = ours+350ms - (cam+100ms)
	if !strings.Contains(sum, "our audio is 250 ms later than our video") {
		t.Fatalf("summary %q", sum)
	}
	s2, _ := newTestSink(t, true)
	if got := s2.srSummaryLocked(); got != "; no camera sender reports" {
		t.Fatalf("summary without reports %q", got)
	}
}

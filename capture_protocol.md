# Capture protocol — one page

**Route 2 (stock tools).** Nothing to install from me. Follow this page literally; if
anything is ambiguous, tell me and I'll fix the page, not the capture.

**Before you start:** lights on · don't tidy up, furniture is expected · don't cover
mirrors or glass, just tell me they're there. Any **one** tier below is a complete
capture — pick on the day.

## The three rules that decide whether this works

1. **Sweep the phone upward at every corner** — tilt up until the ceiling fills the
   screen, hold 2 seconds, come down. Skip this and there is no ceiling height, only
   a wide interval.
2. **Move your feet between photos.** Never stand still and rotate: two photos from
   one spot contain no depth information and cannot be triangulated.
3. **Finish where you started**, and send **original files** — not via WhatsApp,
   Telegram or Slack, which strip the metadata the pipeline reads.

## How to capture

| | **LiDAR** ~90 s/room | **Video** ~60 s/room | **Photo** ~45 s/room |
|---|---|---|---|
| **App** | **Stray Scanner** (App Store, free) | Camera → Video, 1080p/4K 30fps | Camera → Photo, **1×** lens |
| **Device** | iPhone Pro (12 Pro or newer) | iPhone 15 or newer | iPhone 15 or newer |
| **How** | Record. Portrait, chest height. Walk the perimeter **slowly** (~1 step/sec), far wall in view. **Sweep up at all 4 corners.** Return to start, stop. | Start at the doorway, walk the perimeter **slower than feels natural**, phone level. **Pause 2 s at each corner**, tilt up to the ceiling and back. Don't rotate the phone mid-clip. | **One photo per wall**, taken from across the room so the wall and both its corners are in frame. Then 2–3 more from a corner and the doorway. **5–8 per room, each ≥1 m apart.** |
| **Rooms** | Keep recording through doorways — several rooms in one pass is fine | Same; I split the clip into rooms automatically | One folder per room |
| **Hand-off** | Tap recording → **Share** → AirDrop. Send the whole folder unchanged | The `.mov` straight from Photos | AirDrop originals, or export as **"Unmodified Originals"** |

> Walk slowly: blurred frames are discarded before reconstruction — on one real
> 37-second walkthrough that cost 47 of 74 sampled frames. **Minimum 2 photos per
> room**; with one, that room fails outright.

## Handing the files over

One folder, plus a small text file named **`capture.json`** — the only thing you write:

```json
{ "capture_id": "walkin_demo", "tier": "lidar", "declared_rooms": ["living_room"] }
```

`tier` is exactly `lidar`, `photo` or `video`. It decides everything; there is no
command-line flag for it.

**LiDAR / video:** files loose in the folder. **Photo:** one subfolder per room inside
`rooms/` — the folder name becomes the room name in the plan.

```
walkin_demo/                       walkin_demo/
├── capture.json                   ├── capture.json
├── rgb.mp4  odometry.csv          └── rooms/
├── camera_matrix.csv                  ├── living_room/  IMG_0001.jpg …
└── depth/  confidence/                └── kitchen/      IMG_0010.jpg …
```

Images may be `.jpg`, `.heic`, `.png` or `.dng`. Then, one command:

```bash
cozmo run --input walkin_demo --out out/walkin_demo
```

*Optional in `capture.json`, all safe to omit:* `space_id` (same string on two captures
of one space enables the repeatability check), `device`, `operator`, `captured_at`,
`notes`. Any other key is rejected.

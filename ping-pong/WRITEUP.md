# Virtual Ping Pong with a LEGO Double Motor

Courtland Doyle · ME 193

### This is what I did

I made a virtual ping pong game where the **LEGO Double Motor is the paddle**. You hold the hub in your hand and play against a computer opponent on a table drawn on the laptop screen. The laptop webcam watches you, the hub's IMU feels your swing, and the hub's motors rumble in your hand when you hit or miss.

To hit the ball, three things have to be true at the same moment:

1. **Your paddle is in the right place.** MediaPipe pose tracks your wrist (and a YOLO model I trained looks for the hub itself) to place the paddle on the table.
2. **You are actually swinging.** A logistic regression classifier I trained on my own swings reads the hub's IMU and decides "swing" or "not a swing".
3. **The ball is in the hitting zone** at the end of your side of the table.

Hold up an **AprilTag** to pick the level (easy / medium / hard sets the ball speed), hold up the start tag, and **whistle to serve**. Games go to **5 points**, like real ping pong. Your **record for continuous hits** is posted live over MQTT to `ME193/Rogers/Courtland` as a floating point number, and the score of the current game is posted to `ME193/Rogers/Courtland/game`.

*(add video: a full rally, ideally ending in a smash)*

*(add screenshot: the game window mid-rally: camera view with the pose skeleton, the table, and the status bar)*

*(add screenshot: the YOU WIN screen with confetti)*

**How the overall flow looks:**

- **Webcam** → MediaPipe Pose (wrist position), YOLO (where the hub is), AprilTags (level and start). All of this runs on its own background thread so the game never waits on the camera.
- **Hub IMU** (~22 readings per second over Bluetooth) → swing classifier (is this a swing?) + swing power (how hard?)
- **Microphone** → FFT whistle detector (serve)
- **Policy** → a state machine that turns all of that into hits, misses, points and games (see the questions below)
- **Opponent** → a PD controller that chases the ball, with a top speed and a reaction delay
- **Output** → the game window, haptic rumbles on the hub, and MQTT on `test.mosquitto.org`

**Everything I used from class**

| Class topic | Where it is in the game |
| --- | --- |
| LEGO motors | The hub is the paddle; its motors are the haptic feedback |
| IMU | Swing detection (trained classifier) and swing power |
| Pose (MediaPipe) | Wrist position → paddle position on the table |
| Training a classifier (logistic regression, train/test split) | The swing / no-swing detector, trained on my own recorded swings |
| AprilTags | Level select (tags 1/2/3) and start / new game (tag 0) |
| YOLO + Roboflow | A detector I trained to find the hub in the camera image |
| Spectrogram / FFT (whistle) | Whistle to serve |
| PD control | The computer opponent's paddle |
| MQTT | Live record and game score |

**Something new that we didn't do in class: haptic feedback.** The hub's two motors don't drive anything. I flip them back and forth fast, and the kick of the motors reversing is what you feel in your hand. Every event has its own pattern, plus the hub's light and beeper, and the screen shows which one just played so you can learn them:

| Event | What you feel / hear |
| --- | --- |
| Serve | 1 short kick, blue light |
| Hit | 2 sharp kicks, green light |
| Smash | 4 hard kicks, high beep, orange light |
| New record | 3 pulses + triple beep, green pulsing light (once per rally) |
| Miss | long buzz, low beep, red light |
| You win the game | victory tune (C-E-G-C) while the light cycles through colors and the motors "dance" |
| You lose the game | three falling beeps and a slow buzz |

**Other things I added to make it feel like real ping pong:**

- **Swing power:** a harder swing sends a faster return. Anything over 75% power is a **SMASH**.
- **Aiming:** where the ball meets the paddle sets the return angle (like Pong). Hitting off the edge of the paddle sends it to the corner.
- **Games to 5** with a scoreboard, a restart (`n` or tag 0), and a celebration when you win: three waves of confetti, a pulsing YOU WIN banner, and the victory dance on the hub.
- **The rally speeds up** 3% per hit, up to 1.5× the level's speed.

**MQTT topics and message format**

| Topic | What it carries | Example |
| --- | --- | --- |
| `ME193/Rogers/Courtland` | My record (longest streak of continuous hits) as a float, sent every time it goes up | `12.0` |
| `ME193/Rogers/Courtland/game` | The current game to 5, my points first | `3-2`, then `5-3 YOU WIN` |

The game score is on its own subtopic so the record topic only ever carries a plain number, which is what the assignment asked for. Both are sent with **retain**, so anyone who subscribes later immediately gets the latest record instead of waiting for my next one. The record is also saved to a file, so it survives restarting the game.

This is what a subscriber actually received while I played (from `check_score.py`):

```
Listening on test.mosquitto.org / ME193/Rogers/Courtland  (Ctrl+C to stop)
[11:01:43] ME193/Rogers/Courtland: 9.0     <- my saved record, delivered on subscribe (retain)
[11:02:17] ME193/Rogers/Courtland: 9.0     <- the game started and announced it
[11:03:58] ME193/Rogers/Courtland: 10.0    <- beat my record...
[11:04:01] ME193/Rogers/Courtland: 11.0    <- ...one hit later
[11:04:04] ME193/Rogers/Courtland: 12.0    <- new record: 12 hits in a row
```

The 3 seconds between 10, 11 and 12 is one return per hit, so it really is live.

**The levels**

| Level (AprilTag) | Ball speed | Paddle width | Opponent PD gains (Kp / Kd) | Opponent top speed | Opponent reaction delay | Opponent aims away from you |
| --- | --- | --- | --- | --- | --- | --- |
| Easy (tag 1) | 0.45 tables/s | 30% of the table | 18 / 6 | 0.20 | 0.30 s | 20% of the time |
| Medium (tag 2) | 0.70 | 22% | 30 / 8 | 0.32 | 0.20 s | 50% |
| Hard (tag 3) | 1.00 | 16% | 60 / 12 | 0.50 | 0.12 s | 80% |

**It works without the motor.** I didn't have the hub when I started, so I built a simulated paddle: press SPACE and it generates IMU readings shaped like a real swing, and the real classifier still has to recognize them. The rumbles become a screen shake and laptop beeps. There are also 34 automated tests for the game rules, the PD opponent, the swing features, the whistle detector and the MQTT messages, so I could change things without breaking the game.

**Code on GitHub:** `ping-pong` folder in the ME-193 repo

- `game.py`: the main loop and the game window (camera view, table, status bar, celebration)
- `game_logic.py`: the rules and the policy, the ball, and the PD opponent (no camera or hardware, so it can be tested)
- `paddle_hw.py`: the real hub (IMU stream + haptic patterns) and the simulated paddle
- `imu_swing.py`, `collect_swing_data.py`, `train_swing_classifier.py`: record swings, train and test the swing classifier, swing power
- `vision.py`: pose, YOLO, AprilTags, and combining them into one paddle position, on a background thread
- `whistle.py`: whistle-to-serve
- `score.py`, `check_score.py`: MQTT publishing, and a listener to prove it works
- `import_roboflow.py`, `train_hub_yolo.py`: turn the Roboflow export into a dataset and train YOLO on it
- `hub_check.py`: first test with the real hub (connection, IMU numbers, every rumble)
- `make_tags.py`: prints the AprilTags
- **The policy** (`game_logic.py`): what happens every tick while the ball is in play

    ```python
    if ball.toward_player:
        self.opponent.target = 0.5      # opponent drifts back to center
        in_zone = config.CONTACT_ZONE[0] <= ball.y <= config.CONTACT_ZONE[1]
        lined_up = abs(ball.x - paddle_x) <= half
        if in_zone and lined_up and swinging:
            return self._player_hit(now, paddle_x, swing_power)
        if ball.y > config.CONTACT_ZONE[1]:
            return self._player_miss(now, lined_up)
    else:
        # the opponent needs a moment to read your shot before chasing it
        if now >= self._react_at:
            self.opponent.target = ball.landing_x
        if ball.y <= 0.0:
            if abs(self.opponent.x - ball.x) <= OPPONENT_WIDTH / 2:
                self._opponent_return(paddle_x)
            else:
                # PD controller couldn't get there in time: your point
                self.player_points += 1
                return ["winner"] + self._end_point(now)
    ```

- **A hit** (`game_logic.py`): aim from where the ball meets the paddle, speed from swing power

    ```python
    # Like Pong: where the ball meets the paddle sets the return angle.
    offset = (self.ball.x - paddle_x) / (self.params["paddle_width"] / 2)
    to_x = float(np.clip(self.ball.x + 0.35 * offset, 0.05, 0.95))
    # Swing power: a harder swing sends a faster return, which gives the
    # opponent less time to get there.
    slow, fast = config.RETURN_SPEED_RANGE          # 0.8x .. 1.4x
    self.ball.launch(self.ball.x, self.ball.y, to_x, 0.0,
                     self.ball_speed * (slow + (fast - slow) * power))
    ```

- **The PD opponent** (`game_logic.py`)

    ```python
    def update(self, dt, kp, kd, vmax=float("inf")):
        error = self.target - self.x
        accel = kp * error - kd * self.v
        self.v = float(np.clip(self.v + accel * dt, -vmax, vmax))   # top speed
        self.x += self.v * dt
    ```

- **Swing features** (`imu_swing.py`): 0.3 s of IMU readings → 10 numbers for the classifier

    ```python
    acc, gyro = s[:, :3], s[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_dyn = np.abs(acc_mag - GRAVITY_MG)       # how far from 1 g (just holding it)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    return np.array([
        acc_dyn.max(), acc_dyn.mean(), acc_mag.std(),
        np.abs(np.diff(acc_mag)).max(),          # suddenness
        gyro_mag.max(), gyro_mag.mean(), gyro_mag.std(),
        *np.ptp(gyro, axis=0),                   # range of rotation on each axis
    ])
    ```

- **Swing detector + power** (`imu_swing.py`)

    ```python
    self.prob = float(self.model.predict_proba(feats.reshape(1, -1))[0, 1])
    if self.prob >= config.SWING_THRESHOLD and now >= self._cooldown_until:
        if not self.swinging(now):
            self.power = 0.0      # a new swing starts
        self._latched_until = now + config.SWING_LATCH_S
    if self.swinging(now):
        # the classifier often fires before the swing peaks, so keep
        # raising the power estimate for as long as the swing is live
        self.power = max(self.power, swing_power(window))
    ```

- **Whistle detector** (`whistle.py`): one column of a spectrogram, ~20 times a second

    ```python
    spectrum = np.abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
    freqs = np.fft.rfftfreq(len(block), 1 / sample_rate)
    band = (freqs >= 600) & (freqs <= 4000)          # where whistles live
    band_power = spectrum[band]
    peak_i = int(np.argmax(band_power))
    tonality = band_power[peak_i] / np.median(band_power)
    whistle_like = rms >= 0.01 and tonality >= 25    # one tall, narrow peak
    ```

### Questions

**Describe the policy: how does it make decisions?**

The policy is a **rule-based state machine** that sits on top of the learned models. The models do the sensing (where is the paddle, is that a swing, is that a whistle, which tag is that) and the rules decide what happens.

1. **States:** SETUP (show a level tag, then the start tag) → READY (whistle to serve) → RALLY (ball in play) → POINT (short pause after each point) → back to READY, until someone reaches 5 → GAME_OVER (tag 0 or `n` for a new game). AprilTags only count between rallies, so a tag in the background can't change the level mid-point, and a tag has to be seen 5 frames in a row before it counts.
2. **Where is the paddle:** pose gives my wrist, YOLO gives boxes that might be the hub. If a YOLO box is within 15% of the frame of my wrist, I use the box. If not, I use the wrist. The middle 60% of the camera image maps onto the whole table (so I don't have to walk out of frame to reach the edges), it's mirrored so the paddle moves the same way I do, and it's smoothed so it doesn't jitter.
3. **Is it a swing:** every frame, the last 0.3 s of IMU data goes through the classifier. If P(swing) ≥ 0.6, the swing "counts" for the next 0.25 s. After it hits a ball there's a 0.45 s cooldown, so one swing can't hit two balls.
4. **Hit or miss:** when the ball is in the last 18% of my side of the table, it's a **hit** only if the paddle is lined up with the ball **and** I'm swinging. If the ball gets past, it's a **miss**, and the game tells me which part I got wrong ("you were there but didn't swing!" or "paddle wasn't in the right spot!").
5. **Where the return goes:** the return angle comes from where the ball hit the paddle (center = straight, edge = toward the corner), and the return speed comes from swing power: how fast the hub was spinning at the peak of the swing, from 200 deg/s (0%) to 480 deg/s (100%). That maps to 0.8× to 1.4× ball speed. Over 75% is a smash.
6. **The opponent:** a PD controller (`accel = Kp·error − Kd·velocity`) moves its paddle toward where the ball will land. On my first version it **never missed**: in a 10-minute simulation it returned every single ball at every level, so I could never win a point. Real players have a top speed and need a moment to react, so I gave it both. I tuned the speed cap by simulating thousands of shots, until about 1 in 5 angled shots won the point, versus about 1 in 10 shots down the middle. The opponent's own aim is part of the policy too: on hard it hits away from my paddle 80% of the time.
7. **Points and the record:** I win a point when the opponent can't reach my return, it wins one when I miss. My streak counts continuous hits and only a miss resets it, so winning a point keeps it going. Every time the streak beats my record, the new record goes out over MQTT.

**What are the potential limitations of the game?**

- **The swing model only knows my swings, held my way.** The two strongest signals it learned are rotation around the hub's x axis (weight +2.46) and *against* rotation around its z axis (−1.71). So it learned the direction my swing turns the hub, not just "moving fast". If someone holds the hub differently, it may miss their swings. It also only saw 28 seconds of "not swinging" movement, so a big non-swing movement can still sometimes count as a swing.
- **The YOLO hub model is weak.** It was trained on 14 photos (2 more had broken labels) of a hub that only takes up about 5% of the frame, all in the same room. On the 3 validation photos it scored precision 0.57, recall 0.33, mAP50 0.34: it found the hub in 1 of 3. In one photo it was 100% confident about a spot on the wall. That's why the game only trusts a YOLO box close to my wrist and falls back to pose otherwise. In practice, pose does most of the tracking.
- **Left/right only.** The paddle only moves side to side. Pose and the camera can't tell how far forward my hand is, so there's no depth, and there's no spin.
- **The 2D table isn't real physics.** The ball flies in straight lines with a drawn arc. There's no bounce, net or spin, just a fixed "hitting zone".
- **The IMU is slow over Bluetooth.** I asked for a reading every 20 ms but measured about 22 per second, so a 0.3 s window only has 6 or 7 readings. A very fast flick can fall between readings, and power is only an estimate of the true peak.
- **The whistle detector is tuned to my room.** Loud music with a pure tone in it, or a fire alarm, could serve the ball. A quiet whistle in a loud room may not.
- **Haptics compete with the IMU.** The rumble shakes the hub, which is the same sensor that detects swings. The cooldown after each hit covers most of it.
- **Public MQTT broker.** `test.mosquitto.org` has no password, so anyone could publish a fake score to my topic.

**What AI/ML algorithms did you use, and how do they work?**

| Algorithm | How it works (two sentences or less) |
| --- | --- |
| **Logistic regression** (swing detector, trained by me) | It multiplies each of the 10 IMU features by a learned weight, adds them up, and squashes the total into a probability between 0 and 1. Training picks the weights that best separate my recorded swings from my non-swings. |
| **MediaPipe Pose** (pretrained, from Google) | A neural network finds a person in each frame and predicts the location of 33 body keypoints, like wrists and shoulders. I only use the paddle-hand wrist. |
| **YOLOv8** (fine-tuned by me on hub photos) | A convolutional neural network looks at the whole image once and predicts boxes plus a confidence for each object it knows. I started from a model pretrained on everyday objects (transfer learning) and retrained it on one new class, the hub. |
| **AprilTag detection** (OpenCV) | It finds dark square outlines in the image, reads the black/white grid inside as a binary code, and matches it to a known tag ID. The code has error correction, so a slightly blurry tag still reads correctly. |
| **FFT whistle detection** (spectrogram) | Every ~46 ms of audio is turned into a frequency spectrum, one column of a spectrogram. A whistle is a nearly pure tone, so it shows up as one tall, narrow peak, while talking and noise spread across many frequencies. |
| **PD control** (opponent) | It accelerates toward the target in proportion to how far away it is (P), and brakes in proportion to how fast it's already moving (D) so it doesn't overshoot. |

**How good is the swing model? Can you confuse it?**

I recorded **48 swings and 28 seconds of "holding it but not swinging"** with the real hub, then cut them into 0.3 s windows: 418 windows (144 swing, 274 not-swing).

My first test said **97.6%** accuracy, but that number was too good. Each swing makes three overlapping windows, and a random train/test split could put one in training and its near-copy in the test set: a leak. When I kept all windows from the same swing (or the same second of idle movement) on the same side of the split, the honest number was:

| Split | Accuracy | Swings missed | False alarms |
| --- | --- | --- | --- |
| Random split (leaky) | 97.6% | 2 of 29 | 0 of 55 |
| Grouped split (honest) | **94.2%** | 4 of 39 | 0 of 30 |

Confusion matrix (grouped split, rows = true, columns = predicted):

| | Predicted not-swing | Predicted swing |
| --- | --- | --- |
| **True not-swing** | 30 | 0 |
| **True swing** | 4 | 35 |

It errs toward missing a swing rather than inventing one, which is the better mistake in a game. A swing that doesn't count is annoying, but a phantom swing would let you win without trying.

**Ways to confuse it:**

| Test | Result |
| --- | --- |
| Holding the hub differently than when I recorded | Swings get missed more, because it learned the axis my swing rotates around |
| Big non-swing movements (walking, raising the hub fast) | Usually ignored, since the idle recording included them, but a sharp enough jolt can count |
| Very soft swings | Missed: they look like idle movement. The swing threshold is a trade-off between these two |
| The rumble right after a hit | Covered by the 0.45 s cooldown after each hit |

*(add the confusion matrix and the YOLO training charts from `runs/hub/results.png`)*

### Reflection questions

- **What I learned:** how to train a classifier on sensor data I collected myself, not just images. Turning raw IMU readings into features, recording labeled data, and splitting it into train and test. The biggest lesson was that **how you split your data matters**: my first accuracy number was inflated because near-copies of the same swing were in both the training and test sets. I also learned that a few photos of a small object isn't enough for YOLO, and that a PD controller with no limits is a superhuman opponent.
- **What I am most proud of:** the haptic feedback, and that every class topic actually does something in the game instead of being bolted on. You feel a hit, a smash and a miss in your hand, and the hub plays a victory dance when you win. I'm also proud that it works without the hardware: the simulated paddle and the automated tests let me build and test the whole game before I had the motor.
- **What took a while:** getting it to feel right, more than getting it to work. The game kept freezing every now and then until I moved the camera and the vision models onto their own thread. The opponent never missed until I gave it a top speed and a reaction delay, then tuned those by simulating thousands of shots. And the swing power range came from measuring my own recorded swings, not guessing.

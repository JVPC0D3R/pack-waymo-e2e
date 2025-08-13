# Scenario Format

### Train
```
agent/front_left/img: (1079, 972, 3)
agent/front_left/calib/intr: (9,)
agent/front_left/calib/extr: (16,)
agent/front/img: (1079, 972, 3)
agent/front/calib/intr: (9,)
agent/front/calib/extr: (16,)
agent/front_right/img: (1079, 972, 3)
agent/front_right/calib/intr: (9,)
agent/front_right/calib/extr: (16,)
agent/side_left/img: (1079, 972, 3)
agent/side_left/calib/intr: (9,)
agent/side_left/calib/extr: (16,)
agent/side_right/img: (1079, 972, 3)
agent/side_right/calib/intr: (9,)
agent/side_right/calib/extr: (16,)
agent/rear_left/img: (587, 972, 3)
agent/rear_left/calib/intr: (9,)
agent/rear_left/calib/extr: (16,)
agent/rear/img: (551, 972, 3)
agent/rear/calib/intr: (9,)
agent/rear/calib/extr: (16,)
agent/rear_right/img: (587, 972, 3)
agent/rear_right/calib/intr: (9,)
agent/rear_right/calib/extr: (16,)
agent/intent: (1,)
agent/valid: (36,)
agent/vel: (6,)
history/agent/pose: (16,)
agent/pos: (36, 2)
history/agent/pos: (16, 2)
history/agent/vel: (16, 2)
history/agent/acc: (16, 2)
history/agent/valid: (16,)
gt/pos: (20, 3)
```

### Validation
```
agent/front_left/img: (1079, 972, 3)
agent/front_left/calib/intr: (9,)
agent/front_left/calib/extr: (16,)
agent/front/img: (1079, 972, 3)
agent/front/calib/intr: (9,)
agent/front/calib/extr: (16,)
agent/front_right/img: (1079, 972, 3)
agent/front_right/calib/intr: (9,)
agent/front_right/calib/extr: (16,)
agent/side_left/img: (1079, 972, 3)
agent/side_left/calib/intr: (9,)
agent/side_left/calib/extr: (16,)
agent/side_right/img: (1079, 972, 3)
agent/side_right/calib/intr: (9,)
agent/side_right/calib/extr: (16,)
agent/rear_left/img: (587, 972, 3)
agent/rear_left/calib/intr: (9,)
agent/rear_left/calib/extr: (16,)
agent/rear/img: (551, 972, 3)
agent/rear/calib/intr: (9,)
agent/rear/calib/extr: (16,)
agent/rear_right/img: (587, 972, 3)
agent/rear_right/calib/intr: (9,)
agent/rear_right/calib/extr: (16,)
agent/intent: (1,)
agent/valid: (36,)
agent/vel: (6,)
history/agent/pose: (16,)
agent/pos: (36, 2)
history/agent/pos: (16, 2)
history/agent/vel: (16, 2)
history/agent/acc: (16, 2)
history/agent/valid: (16,)
gt/preference_scores: (3,)
gt/pos: (20, 3)
```

### Test
```
agent/front_left/img: (1079, 972, 3)
agent/front_left/calib/intr: (9,)
agent/front_left/calib/extr: (16,)
agent/front/img: (1079, 972, 3)
agent/front/calib/intr: (9,)
agent/front/calib/extr: (16,)
agent/front_right/img: (1079, 972, 3)
agent/front_right/calib/intr: (9,)
agent/front_right/calib/extr: (16,)
agent/side_left/img: (1079, 972, 3)
agent/side_left/calib/intr: (9,)
agent/side_left/calib/extr: (16,)
agent/side_right/img: (1079, 972, 3)
agent/side_right/calib/intr: (9,)
agent/side_right/calib/extr: (16,)
agent/rear_left/img: (587, 972, 3)
agent/rear_left/calib/intr: (9,)
agent/rear_left/calib/extr: (16,)
agent/rear/img: (551, 972, 3)
agent/rear/calib/intr: (9,)
agent/rear/calib/extr: (16,)
agent/rear_right/img: (587, 972, 3)
agent/rear_right/calib/intr: (9,)
agent/rear_right/calib/extr: (16,)
agent/intent: (1,)
agent/vel: (6,)
history/agent/pose: (16,)
history/agent/pos: (16, 2)
history/agent/vel: (16, 2)
history/agent/acc: (16, 2)
history/agent/valid: (16,)
```

# Ego-vehicle intent

```
GO_STRAIGHT = 1
GO_LEFT = 2
GO_RIGHT = 3
```

# Waymo Shard Fields

### Train
```
frame
future_states
past_states
intent
```

### Validation
```
frame
future_states
past_states
intent
preference_trajectories
```

### Test
```
frame
past_states
intent
```
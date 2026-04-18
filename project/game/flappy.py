"""
game/flappy.py — pure game logic, no Qt, no EEG.

Public API:
    game = Game()
    game.jump()              # bird flaps
    alive = game.update()    # advance one tick; False = game over
    game.pipe_data()         # [(x, top_h, bot_y, bot_h), …] for rendering
    game.score               # pipes cleared so far
"""

import random

# ── canvas ────────────────────────────────────────────────────────────────────
CANVAS_W = 360
CANVAS_H = 871

# ── bird ──────────────────────────────────────────────────────────────────────
BIRD_X      = 80      # fixed horizontal position (px from left)
GRAVITY     = 0.5     # px / tick²
JUMP_VEL    = -10.0   # upward velocity on flap

# ── pipes ─────────────────────────────────────────────────────────────────────
PIPE_WIDTH     = 52
PIPE_GAP       = 250   # vertical gap between top and bottom pipe
PIPE_SPEED     = 2.5   # px / tick  (~150 px/s at 60 fps)
PIPE_SPAWN_X   = CANVAS_W + 40
SPAWN_INTERVAL = 150   # ticks between spawns  (~1.6 s at 60 fps)

_GAP_MIN_Y = PIPE_GAP // 2 + 160
_GAP_MAX_Y = CANVAS_H - PIPE_GAP // 2 - 160


# ── primitives ────────────────────────────────────────────────────────────────

class Bird:
    def __init__(self):
        self.y   = float(CANVAS_H * 0.35)   # start ~1/3 from top
        self.vel = 0.0

    def update(self):
        self.vel += GRAVITY
        self.y   += self.vel

    def jump(self):
        self.vel = JUMP_VEL


class Pipe:
    def __init__(self):
        self.x      = float(PIPE_SPAWN_X)
        self.gap_y  = float(random.uniform(_GAP_MIN_Y, _GAP_MAX_Y))
        self.passed = False

    @property
    def top_height(self) -> float:
        return max(0.0, self.gap_y - PIPE_GAP / 2)

    @property
    def bottom_y(self) -> float:
        return self.gap_y + PIPE_GAP / 2

    @property
    def bottom_height(self) -> float:
        return max(0.0, CANVAS_H - self.bottom_y)

    def update(self):
        self.x -= PIPE_SPEED

    def is_offscreen(self) -> bool:
        return self.x + PIPE_WIDTH < 0

    def collides(self, bird_y: float) -> bool:
        """Bird hitbox: ±14 px horizontal, ±10 px vertical around centre."""
        if BIRD_X + 14 < self.x or BIRD_X - 14 > self.x + PIPE_WIDTH:
            return False
        return bird_y - 10 < self.top_height or bird_y + 10 > self.bottom_y


# ── game ──────────────────────────────────────────────────────────────────────

class Game:
    def __init__(self):
        self.bird  = Bird()
        self.pipes: list[Pipe] = []
        self.score = 0
        self.alive = True
        self._ticks_to_spawn = SPAWN_INTERVAL

    # ── public ────────────────────────────────────────────────────────────────

    def jump(self):
        """Make the bird flap. No-op if game is already over."""
        if self.alive:
            self.bird.jump()

    def update(self) -> bool:
        """
        Advance one game tick.
        Returns True while the game is still running, False on death.
        """
        if not self.alive:
            return False

        # Bird physics
        self.bird.update()

        # Death: hit floor
        if self.bird.y >= CANVAS_H:
            self.alive = False
            return False

        # Ceiling clamp (no death at top)
        self.bird.y = max(0.0, self.bird.y)

        # Spawn pipes
        self._ticks_to_spawn -= 1
        if self._ticks_to_spawn <= 0:
            self.pipes.append(Pipe())
            self._ticks_to_spawn = SPAWN_INTERVAL

        # Update pipes — collision + scoring
        for pipe in self.pipes:
            pipe.update()
            if pipe.collides(self.bird.y):
                self.alive = False
                return False
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.passed = True
                self.score += 1

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if not p.is_offscreen()]

        return True

    def pipe_data(self) -> list[tuple]:
        """Render-ready snapshot: [(x, top_height, bottom_y, bottom_height), …]"""
        return [
            (p.x, p.top_height, p.bottom_y, p.bottom_height)
            for p in self.pipes
        ]

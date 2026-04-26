# game/flappy.py
class Bird:
    def __init__(self, y, gravity=0.45, jump=10.5):
        self.y = y
        self.vel = 0
        self.gravity = gravity
        self.jump_strength = jump

    def update(self):
        self.vel += self.gravity
        self.y += self.vel

    def jump(self):
        self.vel = -self.jump_strength

from collections import deque
# import pdb


# TODO  reset, update functions
class Obj:
    def __init__(self, ID, x, y, w, h, feature):
        self.ID = ID
        self.x = deque([x], maxlen=60)
        self.y = deque([y], maxlen=60)
        self.w = deque([w], maxlen=60)
        self.h = deque([h], maxlen=60)
        self.feature = feature
        self.delta_x = 0
        self.delta_y = 0

    def update_loc(self, x, y):  # update w h ~~~
        self.x.appendleft(x)
        self.y.appendleft(y)

    def get_motion(self):
        print('motion')
        if len(self.x) == 2:
            self.delta_x = self.x[0] - self.x[1]
            self.delta_y = self.y[0] - self.y[1]
        elif len(self.x) > 2:
            acc_x = (self.x[0] - self.x[1]) - (self.x[1] - self.x[2])
            acc_y = (self.y[0] - self.y[1]) - (self.y[1] - self.y[2])
            self.delta_x = (self.x[0] - self.x[1]) + 0.2*acc_x
            self.delta_y = (self.y[0] - self.y[1]) + 0.2*acc_y

    def pred_loc(self):
        print('location')
        return self.x[0] + self.delta_x, self.y[0] + self.delta_y


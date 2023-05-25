import math


class Polygon:
    def __init__(self):
        self.points = []
        self.minx = self.miny = math.inf
        self.maxx = self.maxy = -math.inf

    def add_point(self, x, y):
        self.points.append((x, y))
        self._update_boundary(x, y)

    def valid(self):
        """
        # polygon must have at least 3 points and it should not be a line
        TODO check tilt lines
        :return:
        """
        return len(self.points) > 2 and \
               self.maxx > self.minx and \
               self.maxy > self.miny

    def _update_boundary(self, x, y):
        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x)
        self.maxy = max(self.maxy, y)

    def bbox_contain(self, x, y) -> bool:
        """
        Checks if a point is inside (or on the edge of) the bounding box of the polygon
        :param x:
        :param y:
        :return:
        """
        return self.valid() and \
               self.minx <= x <= self.maxx and \
               self.miny <= y <= self.maxy

    def contain(self, x, y) -> bool:
        """
        Check how many times a ray starting from point (x, y) to the right crosses each line segment of polygon
        If that number is odd, then point is inside (or on the edge) the polygon, otherwise outside
        :param x:
        :param y:
        :return:
        """
        if not self.valid():
            print("Polygon is invalid!")
            return False

        if not self.bbox_contain(x, y):
            # print("Point not inside the bounding box!")
            return False

        num = len(self.points)
        inside = False
        i, j = 0, num - 1
        while i != num:
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            if min(yi, yj) <= y <= max(yi, yj):
                # when this line is pure horizontal and point is on this line
                if yi == yj:
                    return True
                # when this line is pure vertical and point is on this line
                elif xi == xj == x:
                    return True
                # when this line is not horizontal, then calculating its slope reciprocal is safe
                elif (xi - xj) / (yi - yj) * (y - yi) + xi > x:
                    inside = not inside
            j = i
            i += 1
        return inside

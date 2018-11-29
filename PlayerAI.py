from random import randint
import random
from BaseAI import BaseAI
import time
import numpy as np
import math as math


class PlayerAI(BaseAI):

    def getMove(self, grid):

        moves = grid.getAvailableMoves()
        max_util = -np.inf
        turn = -1

        for move in moves:

            grid_iter = grid.clone()
            grid_iter.move(move)

            util_iter = self.minimize_ab(grid_iter, -np.inf, np.inf, time.clock(), 0)

            if util_iter >= max_util:
                max_util = util_iter
                turn = move

        return turn

    def maximize_ab(self, grid, alpha, beta, prev, depth):
        # for early parts of game, there are many blank spaces. limiting depth to 4 keeps turn length < 0.2 s
        if (depth >= 4 or not grid.canMove()) or (time.clock() - prev) > 0.04:
            return self.heuristic(grid)

        max_util = -np.inf
        children = grid.getAvailableMoves()
        for child in children:

            tmp_child = grid.clone()
            tmp_child.move(child)

            max_util = max(max_util, self.minimize_ab(tmp_child, alpha, beta, prev, depth + 1))

            if max_util >= beta:
                break

            alpha = max(max_util, alpha)

        return max_util

    def minimize_ab(self, grid, alpha, beta, prev, depth):

        if (depth >= 4 or not grid.canMove()) or (time.clock() - prev) > 0.04:
            return self.heuristic(grid)

        min_util = np.inf

        # randomly pick tile
        x = random.random()
        if x < 0.9:
            tile = 2
        else:
            tile = 4

        slots = grid.getAvailableCells()

        for slot in slots:

            grid_iter = grid.clone()
            grid_iter.insertTile(slot, tile)
            min_util = min(min_util, self.maximize_ab(grid_iter, alpha, beta, prev, depth + 1))

            if min_util <= alpha:
                break

            beta = min(min_util, beta)

        return min_util

    def heuristic(self, grid):

        if not grid.canMove():
            return -np.inf

        spots = len(grid.getAvailableCells())
        max_tile = grid.getMaxTile()
        log_max = self.log2(max_tile)

        return 10.0 * self.gradient(grid) + 3.0 * spots + 0.1 * self.sum_difference(grid) + 2.0 * self.monotonicity(grid) + log_max

    def sum_difference(self, grid):

        # acts as a smoothness heuristic
        import numpy as np

        cum_dif = 0

        for i in range(4):
            for j in range(4):

                sum_iter = np.inf

                # look in the direction that is the smoothest (assuming a 2 when there is a 0)
                if i < 3:
                    sum_iter = min(sum_iter, abs((grid.map[i][j] or 2) - (grid.map[i+1][j] or 2)))

                if j < 3:
                    sum_iter = min(sum_iter, abs((grid.map[i][j] or 2) - (grid.map[i][j+1] or 2)))

                if i > 0:
                    sum_iter = min(sum_iter, abs((grid.map[i][j] or 2) - (grid.map[i-1][j] or 2)))

                if j > 0:
                    sum_iter = min(sum_iter, abs((grid.map[i][j] or 2) - (grid.map[i][j-1] or 2)))

                # return a negative number, numbers that are closer together are good
                cum_dif = cum_dif - sum_iter

        return cum_dif

    def monotonicity(self, grid):

        # combines the row and col monotonicity attributes
        return self.row_monotonicity(grid) + self.col_monotonicity(grid)

    def row_monotonicity(self, grid):

        # side note: would be cool if i made this a generalized method and passed map and map transposed into it
        # reward monotonic rows

        row_mono = [0, 0]

        for i in range(3):

            col = 0
            next_col = col + 1

            while next_col < 4:
                while next_col < 4 and not grid.map[i][next_col]:
                    next_col += 1
                if next_col >= 4:
                    next_col -= 1

                if grid.map[i][col]:
                    cur = self.log2(grid.map[i][col])
                else:
                    cur = 0

                if grid.map[i][next_col]:
                    next = self.log2(grid.map[i][next_col])
                else:
                    next = 0

                if cur > next:
                    row_mono[0] = cur + next
                elif cur < next:
                    row_mono[1] = next - cur

                col = next_col
                next_col = col + 1

        return max(row_mono)

    def col_monotonicity(self, grid):

        # reward monotonic columns

        col_mono = [0, 0]

        for j in range(3):

            row = 0
            next_row = row + 1

            while next_row < 4:
                while next_row < 4 and not grid.map[next_row][j]:
                    next_row += 1
                if next_row >= 4:
                    next_row -= 1

                if grid.map[row][j]:
                    cur = self.log2(grid.map[row][j])
                else:
                    cur = 0

                if grid.map[next_row][j]:
                    next = self.log2(grid.map[next_row][j])
                else:
                    next = 0

                if cur > next:
                    col_mono[0] = cur + next
                elif cur < next:
                    col_mono[1] = next - cur

                row = next_row
                next_row = row + 1

        return max(col_mono)

    def log2(self, x):
        # computes log base 2

        import math
        if x > 0:
            return math.log(x)/math.log(2)
        else:
            return 0

    def gradient(self, grid):

        import numpy as np

        # check which of the 4 matrices returns the best score
        matrices = [
            [[3, 2, 1, 0], [2, 1, 0, -1], [1, 0, -1, -2], [0, -1, -2, -3]],
            [[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, -0]],
            [[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1], [3, 2, 1, 0]],
            [[-3, -2, -1, 0], [-2, -1, 0, 1], [-1, 0, 1, 2], [0, 1, 2, 3]]
        ]

        max_heur = -np.inf

        for i in range(4):

            heur_iter = 0

            for j in range(4):
                for k in range(4):
                    heur_iter += grid.map[j][k] * matrices[i][j][k]

            max_heur = max(heur_iter, max_heur)

        return max_heur





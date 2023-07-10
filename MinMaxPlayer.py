from Board import *
from Player import *

class MinMaxAgent(Player):

    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1

    def __init__(self):
        self.side = None
        self.cache = {}
        super().__init__()

    def new_game(self, side: int):
        if self.side != side:
            self.side = side
            self.cache = {}

    def final_result(self, result: GameResult):
        pass

    def _min(self, board: Board) -> (float, int):
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return self.cache[board_hash]

        min_value = self.DRAW_VALUE
        action = -1

        winner = board.winner
        if winner == self.side:
            min_value = self.WIN_VALUE
            action = -1
        elif winner == board.other_side(self.side):
            min_value = self.LOSS_VALUE
            action = -1
        else:
            empty_params = board.empty_spot_indexes()
            for index in empty_params:
                b = Board(board.state)
                b.move(tuple(index), board.other_side(self.side))

                res, _ = self._max(b)
                if res < min_value or action == -1:
                    min_value = res
                    action = tuple(index)

                    # Shortcut: Can't get better than that, so abort here and return this move
                    if min_value == self.LOSS_VALUE:
                        self.cache[board_hash] = (min_value, action)
                        return min_value, action

                self.cache[board_hash] = (min_value, action)
        return min_value, action

    def _max(self, board: Board) -> (float, int):
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return self.cache[board_hash]

        max_value = self.DRAW_VALUE
        action = -1

        winner = board.winner
        if winner == self.side:
            max_value = self.WIN_VALUE
            action = -1
        elif winner == board.other_side(self.side):
            max_value = self.LOSS_VALUE
            action = -1
        else:
            empty_params = board.empty_spot_indexes()
            for index in empty_params:
                b = Board(board.state)
                b.move(tuple(index), self.side)

                res, _ = self._min(b)
                if res > max_value or action == -1:
                    max_value = res
                    action = tuple(index)

                    # Shortcut: Can't get better than that, so abort here and return this move
                    if max_value == self.WIN_VALUE:
                        self.cache[board_hash] = (max_value, action)
                        return max_value, action

                self.cache[board_hash] = (max_value, action)
        return max_value, action

    def move(self, board: Board) -> (GameResult, bool):
        score, action = self._max(board)
        _, res, finished = board.move(action, self.side)
        return res, finished
import numpy as np
from enum import Enum

class GameResult(Enum):
    """
    Enum to encode different states of the game. A game can be in progress (NOT_FINISHED), lost, won, or draw
    """
    NOT_FINISHED = 0
    NAUGHT_WIN = 1
    CROSS_WIN = 2
    DRAW = 3


EMPTY = 0  # type: int
NAUGHT = 1  # type: int
CROSS = 2  # type: int

PLAYER_1 = 1
PLAYER_2 = 2

BOARD_DIM = 3  # type: int
WINS = {NAUGHT*BOARD_DIM:NAUGHT, CROSS*BOARD_DIM:CROSS}

class Board:
    def __init__(self, s=None):
        if s is None:
            self.state = np.ones((BOARD_DIM,BOARD_DIM))*EMPTY
            self.winner = EMPTY
            self.status = GameResult.NOT_FINISHED
            self.reset()
        else:
            assert type(s)==np.ndarray
            assert (BOARD_DIM,BOARD_DIM)==s.shape
            self.state = s.copy()
            self.winner = EMPTY
            self.status = GameResult.NOT_FINISHED

    def hash_value(self) -> int:
        res = 0
        for i in range(BOARD_DIM):
            for j in range(BOARD_DIM):
                res *= BOARD_DIM
                res += self.state[i,j]
        return res

    def reset(self):
        self.state.fill(EMPTY)
        self.winner = EMPTY
        self.status = GameResult.NOT_FINISHED

    def check_end(self):
        # Check vertical and horizontal sums
        for i in range(BOARD_DIM):
            vertical_sum, horizontal_sum = 0, 0
            if np.sum(self.state[:,i]==EMPTY)==0:
                vertical_sum = np.sum(self.state[:,i])
                if vertical_sum in WINS:
                    self.winner = WINS[vertical_sum]
                    self.status=self.winner
                    return self.state, self.status, True

            if np.sum(self.state[i,:] == EMPTY) == 0:
                horizontal_sum = np.sum(self.state[i,:])
                if horizontal_sum in WINS:
                    self.winner = WINS[horizontal_sum]
                    self.status=self.winner
                    return self.state, self.status, True

        diag_x = np.diagonal(self.state,axis1=0)
        if np.sum(diag_x==EMPTY)==0:
            diag_sum = np.sum(diag_x)
            if diag_sum in WINS:
                self.winner = WINS[diag_sum]
                self.status = self.winner
                return self.state, self.status, True

        diag_y = np.diagonal(self.state,axis1=1,axis2=0)
        if np.sum(diag_y==EMPTY)==0:
            diag_sum = np.sum(diag_y)
            if diag_sum in WINS:
                self.winner = WINS[diag_sum]
                self.status = self.winner
                return self.state, self.status, True

        if np.sum(self.state==EMPTY)==0:
            self.status = GameResult.DRAW
            return self.state, self.status, True

        return self.state, self.status, False

    def move(self, position, side):
        if self.state[position]!=EMPTY:
            print(self)
            print(position)
            print('Illegal move')
            raise ValueError("Invalid move")
        self.state[position] = side

        return self.check_end()

    def random_empty_spot(self):
        assert np.sum(self.state==EMPTY)>0
        empty_params = np.argwhere(self.state == EMPTY)
        return tuple(empty_params[np.random.choice(range(len(empty_params)),1)][0])

    def empty_spot_indexes(self):
        empty_params = np.argwhere(self.state == EMPTY)
        return empty_params

    @staticmethod
    def other_side(side: int) -> int:
        if side == EMPTY:
            raise ValueError("EMPTY has no 'other side'")
        if side == CROSS:
            return NAUGHT
        if side == NAUGHT:
            return CROSS
        raise ValueError("{} is not a valid side".format(side))

    def state_to_char(self, pos, html=False):
        if (self.state[pos]) == EMPTY:
            return '&ensp;' if html else ' '
        if (self.state[pos]) == NAUGHT:
            return 'o'
        return 'x'

    def html_str(self) -> str:
        data = self.state_to_charlist(True)
        html = '<table border="1"><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
        return html

    def state_to_charlist(self, html=False):
        res = []
        for i in range(BOARD_DIM):
            line = [self.state_to_char((i,j), html) for j in range(BOARD_DIM)]
            res.append(line)
        return res

    def __str__(self) -> str:
        board_str = ""
        for i in range(BOARD_DIM):
            for j in range(BOARD_DIM):
                board_str += self.state_to_char((i,j)) + '|'
            board_str = board_str[:-1]
            board_str+="\n"
            if i != (BOARD_DIM-1):
                board_str += "-----\n"
        board_str += "\n"
        return board_str

    def print_board(self):
        print(self)

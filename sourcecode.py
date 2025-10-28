"""
Connect Four Game with Minimax AI
This implementation features a text-based interface where you play against an AI
that uses the minimax algorithm with alpha-beta pruning to make optimal moves.
"""

class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_player = 1  # Human is 1, AI is 2
        self.game_over = False
        self.winner = None
        
    def print_board(self):
        """Print the current game board with visual representation"""
        print("\n" + "=" * 30)
        print("  CONNECT FOUR GAME")
        print("=" * 30)
        
        # Print column numbers for reference
        print("   " + " ".join(str(i) for i in range(self.cols)))
        print("  " + "-" * (self.cols * 2 - 1))
        
        # Print board with symbols
        for row in range(self.rows):
            print("| ", end="")
            for col in range(self.cols):
                if self.board[row][col] == 0:
                    print(".", end=" ")
                elif self.board[row][col] == 1:
                    print("X", end=" ")  # Human
                else:
                    print("O", end=" ")  # AI
            print("|")
        print("  " + "-" * (self.cols * 2 - 1))
        print()
    
    def is_valid_move(self, col):
        """Check if a move is valid (column not full and within bounds)"""
        return 0 <= col < self.cols and self.board[0][col] == 0
    
    def get_valid_moves(self):
        """Return list of valid column moves"""
        return [col for col in range(self.cols) if self.is_valid_move(col)]
    
    def make_move(self, col, player):
        """Make a move in the specified column for the given player"""
        if not self.is_valid_move(col):
            return False
            
        # Find the lowest empty row in the column
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = player
                return True
        return False
    
    def check_winner(self, player):
        """Check if the specified player has won"""
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row][col+i] == player for i in range(4)):
                    return True
        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row+i][col] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row+i][col+i] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(self.board[row+i][col-i] == player for i in range(4)):
                    return True
        
        return False
    
    def is_board_full(self):
        """Check if the board is completely full"""
        return all(self.board[0][col] != 0 for col in range(self.cols))
    
    def evaluate_window(self, window, player):
        """Evaluate a window of 4 consecutive positions"""
        opponent = 3 - player  # 1->2, 2->1
        score = 0
        
        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2
            
        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 4  # Block opponent from winning
        
        return score
    
    def evaluate_board(self, player):
        """Evaluate the entire board for the given player"""
        score = 0
        opponent = 3 - player
        
        # Center column preference (better control)
        center_col = self.cols // 2
        center_array = [self.board[row][center_col] for row in range(self.rows)]
        center_count = center_array.count(player)
        score += center_count * 3
        
        # Evaluate horizontal windows
        for row in range(self.rows):
            for col in range(self.cols - 3):
                window = [self.board[row][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        # Evaluate vertical windows
        for row in range(self.rows - 3):
            for col in range(self.cols):
                window = [self.board[row+i][col] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        # Evaluate diagonal windows (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                window = [self.board[row+i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        # Evaluate diagonal windows (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                window = [self.board[row+i][col-i] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        return score
    
    def get_next_open_row(self, col):
        """Get the next available row in a column"""
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                return row
        return -1
    
    def minimax(self, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning"""
        valid_moves = self.get_valid_moves()
        is_terminal = self.check_winner(1) or self.check_winner(2) or self.is_board_full()
        
        if depth == 0 or is_terminal:
            if self.check_winner(2):  # AI wins
                return None, 1000000000
            elif self.check_winner(1):  # Human wins
                return None, -1000000000
            elif self.is_board_full():  # Draw
                return None, 0
            else:  # Depth limit reached
                return None, self.evaluate_board(2)
        
        if maximizing_player:  # AI's turn
            value = -float('inf')
            best_column = valid_moves[0] if valid_moves else 0
            
            for col in valid_moves:
                # Make move
                row = self.get_next_open_row(col)
                self.board[row][col] = 2
                
                # Recursive call
                new_score = self.minimax(depth-1, alpha, beta, False)[1]
                
                # Undo move
                self.board[row][col] = 0
                
                if new_score > value:
                    value = new_score
                    best_column = col
                
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
                    
            return best_column, value
        
        else:  # Human's turn (minimizing)
            value = float('inf')
            best_column = valid_moves[0] if valid_moves else 0
            
            for col in valid_moves:
                # Make move
                row = self.get_next_open_row(col)
                self.board[row][col] = 1
                
                # Recursive call
                new_score = self.minimax(depth-1, alpha, beta, True)[1]
                
                # Undo move
                self.board[row][col] = 0
                
                if new_score < value:
                    value = new_score
                    best_column = col
                
                beta = min(beta, value)
                if alpha >= beta:
                    break
                    
            return best_column, value
    
    def get_ai_move(self, depth=4):
        """Get the AI's move using minimax"""
        print("AI is thinking...")
        
        # Check for immediate win
        for col in self.get_valid_moves():
            row = self.get_next_open_row(col)
            self.board[row][col] = 2
            if self.check_winner(2):
                self.board[row][col] = 0
                return col
            self.board[row][col] = 0
        
        # Check for blocking human's immediate win
        for col in self.get_valid_moves():
            row = self.get_next_open_row(col)
            self.board[row][col] = 1
            if self.check_winner(1):
                self.board[row][col] = 0
                return col
            self.board[row][col] = 0
        
        # Use minimax for optimal move
        column, value = self.minimax(depth, -float('inf'), float('inf'), True)
        return column
    
    def play_game(self):
        """Main game loop"""
        print("Welcome to Connect Four!")
        print("You are X, AI is O")
        print("Enter column numbers (0-6) to make your moves")
        
        while not self.game_over:
            self.print_board()
            
            if self.current_player == 1:  # Human's turn
                try:
                    col = int(input("Your turn! Enter column (0-6): "))
                    if not self.is_valid_move(col):
                        print("Invalid move! Column is full or out of range.")
                        continue
                    
                    self.make_move(col, 1)
                    
                    if self.check_winner(1):
                        self.print_board()
                        print("🎉 Congratulations! You won!")
                        self.game_over = True
                        self.winner = 1
                    
                except ValueError:
                    print("Please enter a valid number!")
                    continue
                
            else:  # AI's turn
                col = self.get_ai_move()
                self.make_move(col, 2)
                print(f"AI plays in column {col}")
                
                if self.check_winner(2):
                    self.print_board()
                    print("🤖 AI wins! Better luck next time!")
                    self.game_over = True
                    self.winner = 2
            
            # Check for draw
            if self.is_board_full() and not self.game_over:
                self.print_board()
                print("It's a draw!")
                self.game_over = True
            
            # Switch players
            self.current_player = 3 - self.current_player

def main():
    """Main function to start the game"""
    game = ConnectFour()
    game.play_game()

if __name__ == "__main__":
    main()

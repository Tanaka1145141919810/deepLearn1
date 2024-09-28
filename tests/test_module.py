import numpy as np

def main():
    x = np.array([[2,0,5,6],[1,3,3,6],[-1,1,2,1],[1,0,1,3]])
    row,col =x.shape
    E = np.eye(row,col)
    ans = np.linalg.det(x - E)
    print("ans is",ans)
if __name__ == "__main__":
    main()
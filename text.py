from scipy.sparse import coo_matrix, hstack,vstack
A = coo_matrix([[1, 2], [3, 4]])
print(A)
B = coo_matrix([[5,7], [6,8]])
print(hstack([A,B]))
print(hstack([A,B]).toarray())

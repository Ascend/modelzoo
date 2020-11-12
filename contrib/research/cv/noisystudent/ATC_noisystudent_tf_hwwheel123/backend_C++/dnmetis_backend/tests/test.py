import dnmetis_backend as m

print(m.__version__)
assert m.add(1, 2) == 3
print("Test add Ok!")
assert m.subtract(1, 2) == -1
print("Test subtract Ok!")
print("Test Ok!")

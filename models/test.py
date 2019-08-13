def f(play=0, good=1):
    print(play, good)


config = {"play":1,"good":0}

print(f(**config))
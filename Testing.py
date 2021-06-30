from random import randrange

distortions = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                   ('shadows', randrange(-1, 3)), ('highlights', randrange(-3, 1)), ('exposure', randrange(-1, 1))]

print(f"{distortions[0][1]}_{distortions[1][1]}_{distortions[2][1]}_{distortions[3][1]}_{distortions[4][1]}")
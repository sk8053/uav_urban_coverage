import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fig: 반복해서 그려주는 도화지.
fig = plt.figure()
# frames: 매번 그림을 그릴때마다 넘어오는 값들.
frames = []
for i in range(0, 100):
    x = np.random.random(100)
    y = np.random.random(100)
    frames.append((x, y))
# func: 그림을 그려주는 함수,
# frames의 값이 순서대로 넘어옴.
def func(each_frame):
    # each_frame: 만약 func 함수가 i번째 call 되었다면,
    # frames[i]가 넘어옴.
    x, y = each_frame
    plt.scatter(x, y)

# 아래 코드를 실행하면
# fig에 frames에 값에 맞춰서, func대로 그린 그림이
# my_animation에 저장됨.
my_animation = animation.FuncAnimation(fig=fig,
                                        func=func,
                                        frames=frames)
# 저장.
writer = animation.writers['ffmpeg'](fps=25)
my_animation.save(f"test_animation.mp4", writer=writer, dpi=128)
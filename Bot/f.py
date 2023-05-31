import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import datetime

# Generate some initial data
times = [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(60)][::-1]
amounts = [random.randint(0, 100) for _ in range(60)] # replace this with real amounts

fig, ax = plt.subplots()

# Function to update the data
def update(num):
    times.append(datetime.datetime.now())  # add new time
    times.pop(0)  # remove oldest time
    amounts.append(random.randint(0, 100))  # add new amount
    amounts.pop(0)  # remove oldest amount

    ax.clear()
    ax.plot(times, amounts)
    plt.gcf().autofmt_xdate()  # rotates the x-axis labels (dates) if they overlap

ani = animation.FuncAnimation(fig, update, frames=range(60), repeat=True)

plt.title('Time vs Amount')
plt.xlabel('Time')
plt.ylabel('Amount')

plt.show()

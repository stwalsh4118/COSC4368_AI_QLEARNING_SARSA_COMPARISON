import numpy as np
import matplotlib.pyplot as plt
import random as rand
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

pickUps = {(4, 2), (3, 5)}
dropOffs = {(1, 1), (1, 5), (3, 3), (5, 5)}
hasBlock = False


def initQtable():
    q_table = {
        'n': {
            '1,1': '-',
            '1,2': '-',
            '1,3': '-',
            '1,4': '-',
            '1,5': '-',
            '2,1': 0,
            '2,2': 0,
            '2,3': 0,
            '2,4': 0,
            '2,5': 0,
            '3,1': 0,
            '3,2': 0,
            '3,3': 0,
            '3,4': 0,
            '3,5': 0,
            '4,1': 0,
            '4,2': 0,
            '4,3': 0,
            '4,4': 0,
            '4,5': 0,
            '5,1': 0,
            '5,2': 0,
            '5,3': 0,
            '5,4': 0,
            '5,5': 0
        },
        's': {
            '1,1': 0,
            '1,2': 0,
            '1,3': 0,
            '1,4': 0,
            '1,5': 0,
            '2,1': 0,
            '2,2': 0,
            '2,3': 0,
            '2,4': 0,
            '2,5': 0,
            '3,1': 0,
            '3,2': 0,
            '3,3': 0,
            '3,4': 0,
            '3,5': 0,
            '4,1': 0,
            '4,2': 0,
            '4,3': 0,
            '4,4': 0,
            '4,5': 0,
            '5,1': '-',
            '5,2': '-',
            '5,3': '-',
            '5,4': '-',
            '5,5': '-'
        },
        'e': {
            '1,1': 0,
            '1,2': 0,
            '1,3': 0,
            '1,4': 0,
            '1,5': '-',
            '2,1': 0,
            '2,2': 0,
            '2,3': 0,
            '2,4': 0,
            '2,5': '-',
            '3,1': 0,
            '3,2': 0,
            '3,3': 0,
            '3,4': 0,
            '3,5': '-',
            '4,1': 0,
            '4,2': 0,
            '4,3': 0,
            '4,4': 0,
            '4,5': '-',
            '5,1': 0,
            '5,2': 0,
            '5,3': 0,
            '5,4': 0,
            '5,5': '-'
        },
        'w': {
            '1,1': '-',
            '1,2': 0,
            '1,3': 0,
            '1,4': 0,
            '1,5': 0,
            '2,1': '-',
            '2,2': 0,
            '2,3': 0,
            '2,4': 0,
            '2,5': 0,
            '3,1': '-',
            '3,2': 0,
            '3,3': 0,
            '3,4': 0,
            '3,5': 0,
            '4,1': '-',
            '4,2': 0,
            '4,3': 0,
            '4,4': 0,
            '4,5': 0,
            '5,1': '-',
            '5,2': 0,
            '5,3': 0,
            '5,4': 0,
            '5,5': 0
        },
        'p': {
            '1,1': '-',
            '1,2': '-',
            '1,3': '-',
            '1,4': '-',
            '1,5': '-',
            '2,1': '-',
            '2,2': '-',
            '2,3': '-',
            '2,4': '-',
            '2,5': '-',
            '3,1': '-',
            '3,2': '-',
            '3,3': '-',
            '3,4': '-',
            '3,5': 0,
            '4,1': '-',
            '4,2': 0,
            '4,3': '-',
            '4,4': '-',
            '4,5': '-',
            '5,1': '-',
            '5,2': '-',
            '5,3': '-',
            '5,4': '-',
            '5,5': '-'
        },
        'd': {
            '1,1': 0,
            '1,2': '-',
            '1,3': '-',
            '1,4': '-',
            '1,5': 0,
            '2,1': '-',
            '2,2': '-',
            '2,3': '-',
            '2,4': '-',
            '2,5': '-',
            '3,1': '-',
            '3,2': '-',
            '3,3': 0,
            '3,4': '-',
            '3,5': '-',
            '4,1': '-',
            '4,2': '-',
            '4,3': '-',
            '4,4': '-',
            '4,5': '-',
            '5,1': '-',
            '5,2': '-',
            '5,3': '-',
            '5,4': '-',
            '5,5': 0
        }
    }

    q_table = pd.DataFrame.from_dict(q_table, orient='columns', dtype=None)

    return q_table


q_table = initQtable()


def get_key(val, state):
    for action in q_table.keys():
        if q_table[action][state] == val:
            return action

    return "Whoops"


def initWorld(pickUpsList, dropOffsList):
    state = np.zeros(25).reshape(5, 5)
    for x in range(5):
        for y in range(5):
            state[x][y] = -1
    for pu in pickUpsList:
        state[pu[0] - 1][pu[1] - 1] = 8
    for do in dropOffsList:
        state[do[0] - 1][do[1] - 1] = 0
    return state


def applicableOperators(state, world):
    aplop = []
    for action in q_table.keys():
        if q_table[action][state] != '-':
            # if drop off is full
            if action == 'd' and world[int(state.split(',')[0]) - 1][int(state.split(',')[1]) - 1] == 4:
                # print("drop off full")
                pass
            # if actor doesn't have block
            elif action == 'd' and not hasBlock:
                # print("cannot drop off due to having block in possession")
                pass
            # if pick up is empty
            elif action == 'p' and world[int(state.split(',')[0]) - 1][int(state.split(',')[1]) - 1] == 0:
                # print("pick up empty")
                pass
            # if actor already has a block
            elif action == 'p' and hasBlock:
                # print("cannot pick up due to block in possession")
                pass
            else:
                aplop.append(action)
    return aplop


def applyOperator(state, action, world):
    statePrime = ''
    global hasBlock
    if action == 'n':
        statePrime = str(int(state.split(',')[0]) - 1)
        statePrime += ","
        statePrime += state.split(',')[1]
    elif action == 's':
        statePrime = str(int(state.split(',')[0]) + 1)
        statePrime += ","
        statePrime += state.split(',')[1]
    elif action == 'e':
        statePrime = state.split(',')[0]
        statePrime += ","
        statePrime += str(int(state.split(',')[1]) + 1)
    elif action == 'w':
        statePrime = state.split(',')[0]
        statePrime += ","
        statePrime += str(int(state.split(',')[1]) - 1)
    elif action == 'p':
        world[int(state.split(',')[0]) - 1][int(state.split(',')[1]) - 1] -= 1
        statePrime = state
        # print("picking up at ", state)
        hasBlock = True
    elif action == 'd':
        world[int(state.split(',')[0]) - 1][int(state.split(',')[1]) - 1] += 1
        hasBlock = False
        statePrime = state
        # print("placing down at ", state)

    return statePrime


def PRandom(state, world):
    aplops = applicableOperators(state, world)
    action = ''
    for pd in aplops:
        if pd == 'p':
            action = 'p'
        elif pd == 'd':
            action = 'd'
    if action == '':
        action = aplops[rand.randrange(0, len(aplops))]
    return action


def PExploit(state, world):
    aplops = applicableOperators(state, world)
    action = ''
    for pd in aplops:
        if pd == 'p':
            action = 'p'
        elif pd == 'd':
            action = 'd'
    if action == '':
        qPrimeList = []
        for a in aplops:
            qPrimeList.append(q_table[a][state])
        rand.shuffle(qPrimeList)
        QMax = max(qPrimeList)
        isDif = False
        action = get_key(QMax, state)
        while not isDif:
            randaction = aplops[rand.randrange(0, len(aplops))]
            if action != randaction:
                isDif = True
        randselect = rand.randrange(0, 10)
        if randselect >= 2:
            pass
        else:
            action = randaction
    return action


def PGreedy(state, world):
    aplops = applicableOperators(state, world)
    action = ''
    for pd in aplops:
        if pd == 'p':
            action = 'p'
        elif pd == 'd':
            action = 'd'
    if action == '':
        qPrimeList = []
        for a in aplops:
            qPrimeList.append(q_table[a][state])
        rand.shuffle(qPrimeList)
        QMax = max(qPrimeList)
        isDif = False
        action = get_key(QMax, state)
    return action


def QLearn(state, statePrime, action, reward, alpha, gamma, PDWorld):
    aplops = applicableOperators(statePrime, PDWorld)
    qPrimeList = []
    for aprime in aplops:
        # print("qvalue at ", statePrime, " taking action " , aprime, " " ,q_table[aprime][statePrime])
        qPrimeList.append(q_table[aprime][statePrime])
    QMax = max(qPrimeList)
    # print("QMax ", QMax," Prev Q Value ", q_table[action][state]," New Q value for ", state, " ", round(((1 - alpha) * q_table[action][state]) + (alpha * (reward + (gamma * QMax))),10))
    q_table[action][state] = round(((1 - alpha) * q_table[action][state]) + (alpha * (reward + (gamma * QMax))), 10)
    return


def SARSALearn(state, statePrime, action, actionPrime, reward, alpha, gamma, PDWorld):
    q_table[action][state] = round((q_table[action][state]) + alpha * (
                reward + gamma * (q_table[actionPrime][statePrime]) - q_table[action][state]), 10)
    return


def checkTerminal(PDWorld, pickUpsList, dropOffsList):
    isTerminal = True
    for pu in pickUpsList:
        # print(str(pu[0]) + "," + str(pu[1]) + " " + str(PDWorld[pu[0]-1][pu[1]-1]))
        if PDWorld[pu[0] - 1][pu[1] - 1] != 0:
            isTerminal = False
    for do in dropOffsList:
        # print(str(do[0]) + "," + str(do[1]) + " " + str(PDWorld[do[0]-1][do[1]-1]))
        if PDWorld[do[0] - 1][do[1] - 1] != 4:
            isTerminal = False
    # print("---------------------------")
    return isTerminal


def visualizeQTable():
    fig, ax = plt.subplots(1, 1)
    plt.title('The Q Values')
    visu = pd.DataFrame(np.zeros((15, 15)))
    visu = visu.replace(0, '-')
    qvisu = q_table.replace('-', 0)
    patches = np.array([])
    mask = np.array(np.zeros((15, 15)))  # we use this to mask out the corners of the 3x3 squares

    patches_colors = []

    for x in range(15):
        for y in range(15):
            # north
            if y % 3 == 1 and x % 3 == 0:
                state = str(int((x + 3) / 3)) + "," + str(int((y + 2) / 3))
                visu[y][x] = qvisu['n'][state]
            # west
            elif y % 3 == 0 and x % 3 == 1:
                state = str(int((x + 2) / 3)) + "," + str(int((y + 3) / 3))
                visu[y][x] = qvisu['w'][state]
            # south
            elif y % 3 == 1 and x % 3 == 2:
                state = str(int((x + 1) / 3)) + "," + str(int((y + 2) / 3))
                visu[y][x] = qvisu['s'][state]
            # east
            elif y % 3 == 2 and x % 3 == 1:
                state = str(int((x + 2) / 3)) + "," + str(int((y + 1) / 3))
                visu[y][x] = qvisu['e'][state]
            elif y % 3 == 1 and x % 3 == 1:
                state = str(int((x + 2) / 3)) + "," + str(int((y + 2) / 3))
                if q_table['p'][state] != '-':
                    visu[y][x] = qvisu['p'][state]
                elif q_table['d'][state] != '-':
                    visu[y][x] = qvisu['d'][state]
                patch = q_edges(xy=(x, y), sidelen=3)
                patches = np.append(patches, patch)
            else:
                mask[y, x] = True  # hides the corners

    visu = visu.replace('-', 0)
    ax = sns.heatmap(visu, ax=ax, square=True,
                     annot=True, linewidths=.5,
                     center=0, mask=mask,
                     cmap=sns.diverging_palette(0, 120, l=50, center="dark", as_cmap=True))

    low = ax.collections[0].colorbar.vmin
    width = ax.collections[0].colorbar.vmax - ax.collections[0].colorbar.vmin
    color_map = ax.collections[0].cmap.colors

    for i in range(len(patches)):  # patches are ordered from left->right, top->bottom, and clockwise around the point
        # maps patches to the visu table
        x = 1 + 3 * (i // 20) + (i % 4 == 1 and 1 or i % 4 == 3 and -1 or 0)
        y = 1 + 3 * ((i % 20) // 4) + (i % 4 == 2 and 1 or i % 4 == 0 and -1 or 0)
        color = color_map[int((visu[x][y] - low) / width * len(color_map))]  # reference the color table
        patches_colors.append(color)
    p = PatchCollection(patches)
    p.set_linewidth(0.5)
    p.set_edgecolor((1, 1, 1, 1))
    p.set_facecolor(patches_colors)
    ax.add_collection(p)

    # attempting to maximize window for readability
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    return


# Helper function that generates trapezoids for the each tile
def q_edges(xy=(0, 0), sidelen=1):
    patches = []
    origin = np.array([0.5, 0.5])  # used for rotating
    new_origin = np.array(xy) + 0.5 - sidelen / 2
    trapezoid = np.array(((0, 0), (1, 0), (2 / 3, 1 / 3), (1 / 3, 1 / 3)))
    trapezoid -= origin
    rotation_matrix = np.array([[0, 1], [-1, 0]])  # 90 degrees CLOCKWISE
    for i in range(4):
        # offset trapezoid, scale, and move it to the new origin
        shape = Polygon((trapezoid + origin) * sidelen + new_origin)
        patches.append(shape)
        trapezoid = trapezoid @ rotation_matrix  # rotate the trapezoid 90 degrees CLOCKWISE
    return patches


def visualizeWorld(PDWorld):
    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=PDWorld, loc="center")
    plt.show()
    return


def EXPERIMENT1A(numSteps, alpha, gamma, pu, do):
    totalSteps = 0
    while True:
        state = '5,1'
        PDWorld = initWorld(pu, do)
        if totalSteps < 500:
            action = PRandom(state, PDWorld)
        else:
            action = PRandom(state, PDWorld)
        currentSteps = 0
        while True:
            if totalSteps >= steps:
                print("Terminated due to step count")
                visualizeQTable()
                visualizeWorld(PDWorld)
                return
            if checkTerminal(PDWorld, pu, do):
                print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                visualizeQTable()
                visualizeWorld(PDWorld)
                break
            statePrime = applyOperator(state, action, PDWorld)
            # print("taking action ", action)
            reward = 0
            if action == 'p' or action == 'd':
                reward = 13
            else:
                reward = -1
            if totalSteps < 500:
                actionPrime = PRandom(statePrime, PDWorld)
            else:
                actionPrime = PRandom(statePrime, PDWorld)

            QLearn(state, statePrime, action, reward, alpha, gamma, PDWorld)
            # visualizeQTable()
            # print("current state :", state)
            state = statePrime
            # print("action taken from current state :", action)
            action = actionPrime
            # print(q_table)
            # print("curr state:",state,"block state:",hasBlock)
            totalSteps += 1
            currentSteps += 1

            if totalSteps == 500:
                print("Steps completed:", totalSteps, " steps. ", numSteps - totalSteps,
                      " steps left. Now using next policy.")
                visualizeQTable()
                visualizeWorld(PDWorld)

    return


a = .3
g = .5
steps = 10000
stepsTaken = []
allRewards = []


def EXPERIMENT1A(numSteps, alpha, gamma, pu, do):
    totalSteps = 0
    while True:
        state = '5,1'
        PDWorld = initWorld(pu, do)
        if totalSteps < 500:
            action = PRandom(state, PDWorld)
        else:
            action = PRandom(state, PDWorld)
        currentSteps = 0
        currentReward = 0
        while True:
            if totalSteps >= numSteps:
                print("Terminated due to step count")
                print(stepsTaken)
                print(allRewards)
                print("Total terminal states reached:", len(stepsTaken))
                print("average step count to completion:", np.mean(stepsTaken))
                print("average final reward count:", np.mean(allRewards))
                visualizeQTable()
                visualizeWorld(PDWorld)
                return
            if checkTerminal(PDWorld, pu, do):
                print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                # visualizeQTable()
                # visualizeWorld(PDWorld)
                stepsTaken.append(currentSteps)
                allRewards.append(currentReward)
                break
            statePrime = applyOperator(state, action, PDWorld)
            # print("taking action ", action)
            reward = 0
            if action == 'p' or action == 'd':
                reward = 13
            else:
                reward = -1
            if totalSteps < 500:
                actionPrime = PRandom(statePrime, PDWorld)
            else:
                actionPrime = PRandom(statePrime, PDWorld)

            QLearn(state, statePrime, action, reward, alpha, gamma, PDWorld)
            # visualizeQTable()
            # print("current state :", state)
            state = statePrime
            # print("action taken from current state :", action)
            action = actionPrime
            # print(q_table)
            print("curr state:", state, "block state:", hasBlock)
            totalSteps += 1
            currentSteps += 1
            currentReward += reward
            if totalSteps == 500:
                print("Steps completed:", totalSteps, " steps. ", numSteps - totalSteps,
                      " steps left. Now using next policy.")
                print("Reward count currently at:", currentReward)
                visualizeQTable()
                visualizeWorld(PDWorld)

    return


def EXPERIMENT1B(numSteps, alpha, gamma, pu, do):
    totalSteps = 0
    while True:
        state = '5,1'
        PDWorld = initWorld(pu, do)
        if totalSteps < 500:
            action = PRandom(state, PDWorld)
        else:
            action = PRandom(state, PDWorld)
        currentSteps = 0
        currentReward = 0
        while True:
            if totalSteps >= numSteps:
                print("Terminated due to step count")
                print(stepsTaken)
                print(allRewards)
                print("Total terminal states reached:", len(stepsTaken))
                print("average step count to completion:", np.mean(stepsTaken))
                print("average final reward count:", np.mean(allRewards))
                visualizeQTable()
                visualizeWorld(PDWorld)
                return
            if checkTerminal(PDWorld, pu, do):
                print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                # visualizeQTable()
                # visualizeWorld(PDWorld)
                stepsTaken.append(currentSteps)
                allRewards.append(currentReward)
                break
            statePrime = applyOperator(state, action, PDWorld)
            # print("taking action ", action)
            reward = 0
            if action == 'p' or action == 'd':
                reward = 13
            else:
                reward = -1
            if totalSteps < 500:
                actionPrime = PRandom(statePrime, PDWorld)
            else:
                actionPrime = PGreedy(statePrime, PDWorld)

            QLearn(state, statePrime, action, reward, alpha, gamma, PDWorld)
            # visualizeQTable()
            # print("current state :", state)
            state = statePrime
            # print("action taken from current state :", action)
            action = actionPrime
            # print(q_table)
            print("curr state:", state, "block state:", hasBlock)
            totalSteps += 1
            currentSteps += 1
            currentReward += reward
            if totalSteps == 500:
                print("Steps completed:", totalSteps, " steps. ", numSteps - totalSteps,
                      " steps left. Now using next policy.")
                print("Reward count currently at:", currentReward)
                visualizeQTable()
                visualizeWorld(PDWorld)

    return


def EXPERIMENT1C(numSteps, alpha, gamma, pu, do):
    totalSteps = 0
    while True:
        state = '5,1'
        PDWorld = initWorld(pu, do)
        if totalSteps < 500:
            action = PRandom(state, PDWorld)
        else:
            action = PRandom(state, PDWorld)
        currentSteps = 0
        currentReward = 0
        while True:
            if totalSteps >= numSteps:
                print("Terminated due to step count")
                print(stepsTaken)
                print(allRewards)
                print("Total terminal states reached:", len(stepsTaken))
                print("average step count to completion:", np.mean(stepsTaken))
                print("average final reward count:", np.mean(allRewards))
                visualizeQTable()
                visualizeWorld(PDWorld)
                return
            if checkTerminal(PDWorld, pu, do):
                print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                # visualizeQTable()
                # visualizeWorld(PDWorld)
                stepsTaken.append(currentSteps)
                allRewards.append(currentReward)
                break
            statePrime = applyOperator(state, action, PDWorld)
            # print("taking action ", action)
            reward = 0
            if action == 'p' or action == 'd':
                reward = 13
            else:
                reward = -1
            if totalSteps < 500:
                actionPrime = PRandom(statePrime, PDWorld)
            else:
                actionPrime = PExploit(statePrime, PDWorld)

            QLearn(state, statePrime, action, reward, alpha, gamma, PDWorld)
            # visualizeQTable()
            # print("current state :", state)
            state = statePrime
            # print("action taken from current state :", action)
            action = actionPrime
            # print(q_table)
            print("curr state:", state, "block state:", hasBlock)
            totalSteps += 1
            currentSteps += 1
            currentReward += reward
            if totalSteps == 500:
                print("Steps completed:", totalSteps, " steps. ", numSteps - totalSteps,
                      " steps left. Now using next policy.")
                print("Reward count currently at:", currentReward)
                visualizeQTable()
                visualizeWorld(PDWorld)

    return


a = .3
g = .5
steps = 6000
stepsTaken = []
allRewards = []


def EXPERIMENT2(numSteps, alpha, gamma, pu, do):
    totalSteps = 0
    notifiedFirstDropOff = False
    while True:
        state = '5,1'
        PDWorld = initWorld(pu, do)
        if totalSteps < 500:
            action = PRandom(state, PDWorld)
        else:
            action = PExploit(state, PDWorld)
        currentSteps = 0
        currentReward = 0
        while True:
            if totalSteps >= steps:
                print("Terminated due to step count")
                print(stepsTaken)
                print(allRewards)
                print("Total terminal states reached:", len(stepsTaken))
                print("average step count to completion:", np.mean(stepsTaken))
                print("average final reward count:", np.mean(allRewards))
                visualizeQTable()
                visualizeWorld(PDWorld)
                return
            if checkTerminal(PDWorld, pu, do):
                print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                # visualizeQTable()
                # visualizeWorld(PDWorld)
                stepsTaken.append(currentSteps)
                allRewards.append(currentReward)
                break
            statePrime = applyOperator(state, action, PDWorld)
            # print("taking action ", action)
            reward = 0
            if action == 'p' or action == 'd':
                reward = 13
            else:
                reward = -1
            if totalSteps < 500:
                actionPrime = PRandom(statePrime, PDWorld)
            else:
                actionPrime = PExploit(statePrime, PDWorld)

            SARSALearn(state, statePrime, action, actionPrime, reward, alpha, gamma, PDWorld)
            # visualizeQTable()
            # print("current state :", state)
            state = statePrime
            # print("action taken from current state :", action)
            action = actionPrime
            # print(q_table)
            # print("curr state:", state, "block state:", hasBlock)
            totalSteps += 1
            currentSteps += 1
            currentReward += reward
            if not notifiedFirstDropOff:
                for dropoff in do:
                    if PDWorld[dropoff[0] - 1][dropoff[1] - 1] == 4:
                        notifiedFirstDropOff = True
                        print("Filled the first drop off location at %s steps, %s steps remaining"
                              % (totalSteps, numSteps - totalSteps))
                        print("Reward count currently at:", currentReward)
                        visualizeQTable()
                        visualizeWorld(PDWorld)
                        break
            if totalSteps == 500:
                print("Steps completed:", totalSteps, " steps. ", numSteps - totalSteps,
                      " steps left. Now using next policy.")
                print("Reward count currently at:", currentReward)
                visualizeQTable()
                visualizeWorld(PDWorld)

    return


def EXPERIMENT4(numSteps, alpha, gamma, pu, do):
    totalSteps = 0
    numEpisodes = 0
    while True:
        state = '5,1'
        if numEpisodes <= 2:
            PDWorld = initWorld(pu, do)
            if totalSteps < 500:
                action = PRandom(state, PDWorld)
            else:
                action = PExploit(state, PDWorld)
        else:
            pickUps = {(3, 1), (1, 3)}
            PDWorld = initWorld(pickUps, do)
            if numEpisodes == 3:
                q_table['p']['3,1'] = 0
                q_table['p']['1,3'] = 0
                q_table['p']['4,2'] = '-'
                q_table['p']['3,5'] = '-'
                print("changing pick up spots")
                visualizeWorld(PDWorld)
            if totalSteps < 500:
                action = PRandom(state, PDWorld)
            else:
                action = PExploit(state, PDWorld)

        currentSteps = 0
        currentReward = 0
        while True:
            if totalSteps >= steps:
                print("Terminated due to step count")
                print(stepsTaken)
                print(allRewards)
                print("Total terminal states reached:", len(stepsTaken))
                print("average step count to completion:", np.mean(stepsTaken))
                print("average final reward count:", np.mean(allRewards))
                visualizeQTable()
                visualizeWorld(PDWorld)
                return
            if numEpisodes <= 2:
                if checkTerminal(PDWorld, pu, do):
                    print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                    visualizeQTable()
                    visualizeWorld(PDWorld)
                    stepsTaken.append(currentSteps)
                    allRewards.append(currentReward)
                    numEpisodes += 1
                    break
            else:
                pus = {(3, 1), (1, 3)}
                if checkTerminal(PDWorld, pus, do):
                    print("Terminal State Reached in ", currentSteps, " steps. ", numSteps - totalSteps, " steps left")
                    visualizeQTable()
                    visualizeWorld(PDWorld)
                    stepsTaken.append(currentSteps)
                    allRewards.append(currentReward)
                    numEpisodes += 1
                    break
            statePrime = applyOperator(state, action, PDWorld)
            # print("taking action ", action)
            reward = 0
            if action == 'p' or action == 'd':
                reward = 13
            else:
                reward = -1
            if totalSteps < 500:
                actionPrime = PRandom(statePrime, PDWorld)
            else:
                actionPrime = PExploit(statePrime, PDWorld)

            SARSALearn(state, statePrime, action, actionPrime, reward, alpha, gamma, PDWorld)
            # visualizeQTable()
            # print("current state :", state)
            state = statePrime
            # print("action taken from current state :", action)
            action = actionPrime
            # print(q_table)
            # print("curr state:", state, "block state:", hasBlock)
            totalSteps += 1
            currentSteps += 1
            currentReward += reward
            if totalSteps == 500:
                print("Steps completed:", totalSteps, " steps. ", numSteps - totalSteps,
                      " steps left. Now using next policy.")
                print("Reward count currently at:", currentReward)
                visualizeQTable()
                visualizeWorld(PDWorld)

    return


# EXPERIMENT2(steps, a, g, pickUps, dropOffs)

# EXPERIMENT3A a = .15
# a = .15
# EXPERIMENT2(steps, a, g, pickUps, dropOffs)

# EXPERIMENT3B a = .15
# a = .45
# EXPERIMENT2(steps, a, g, pickUps, dropOffs)

steps = 10000
a = .3
g = .5
EXPERIMENT2(steps, a, g, pickUps, dropOffs)








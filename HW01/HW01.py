import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns


def mapping():
    tree = ET.parse('SampleRoom.xml')
    rows = tree.findall('row')
    Map = []
    for row in rows:
        cells = []
        for cell in row:
            cells.append(cell.text)
        Map.append(cells)
    return Map

def graphical_view(_Map) :
    Numerical_Map = []
    for i in _Map :
        row = []
        for j in i :
            if j == 'obstacle' :
                row.append(0.8)
            elif j == 'robot' :
                row.append(0)
            elif j == 'Battery' :
                row.append(0.35)
            else :
                row.append(1)
        Numerical_Map.append(row)
    ax = sns.heatmap( Numerical_Map , linewidth = 0.5 , linecolor = 'black' , cmap = 'cubehelix' , cbar = False)
    plt.title( "Ajent path finding with A* algorithm" )
    plt.ion()
    plt.show()
    plt.pause(1)



def move(direction_list):
    [x,y] = get_bot_pos(Map)
    c = 0
    blocked_flag = 0
    while (direction_list != [] and  blocked_flag == 0):
        direction = direction_list[0]
        if direction == 'left' and y-1>=0 :
            visited_Map[x][y-1] = Map[x][y-1]
            if Map[x][y-1] == 'obstacle':
                graphical_view(Map)
                blocked_flag = 1
            else :
                Map[x][y] = 'empty'
                Map[x][y-1] = 'robot'
                graphical_view(Map)
                visited_Map[x][y] = 'empty'
                visited_Map[x][y-1] = 'robot'
                if (Map[x][y-1]=='Battery'):
                    steps = steps + c + 1
                    break
            y = y - 1

        elif direction == 'right' and y+1 <len(Map[0]) :
            visited_Map[x][y+1] = Map[x][y+1]
            if Map[x][y+1] == 'obstacle':
                graphical_view(Map)
                blocked_flag = 1
            else :
                Map[x][y] = 'empty'
                Map[x][y+1] = 'robot'
                graphical_view(Map)
                visited_Map[x][y] = 'empty'
                visited_Map[x][y+1] = 'robot'
                if (Map[x][y+1]=='Battery'):
                    steps = steps + c + 1
                    break
            y = y + 1

        elif direction == 'up' and x-1>=0 :
            visited_Map[x-1][y] = Map[x-1][y]
            if Map[x-1][y] == 'obstacle':
                graphical_view(Map)
                blocked_flag = 1
            else :
                Map[x][y] = 'empty'
                Map[x-1][y] = 'robot'
                graphical_view(Map)
                visited_Map[x][y] = 'empty'
                visited_Map[x-1][y] = 'robot'
                if (Map[x-1][y]=='Battery'):
                    steps = steps + c + 1
                    break
            x = x - 1

        elif direction == 'down' and x+1<len(Map) :
            visited_Map[x+1][y] = Map[x+1][y]
            if Map[x+1][y] == 'obstacle':
                graphical_view(Map)
                blocked_flag = 1
            else :
                Map[x][y] = 'empty'
                Map[x+1][y] = 'robot'
                graphical_view(Map)
                visited_Map[x][y] = 'empty'
                visited_Map[x+1][y] = 'robot'
                if Map[x+1][y]=='Battery':
                    steps = steps + c + 1
                    break
            x = x + 1

        if blocked_flag == 1 :
            A_star(visited_Map)
            break
        direction_list.pop(0)
        c = c + 1


steps = 0
def A_star (visited_Map):
    [x,y] = get_bot_pos(Map)
    Heuristic = [[None for _ in range(len(visited_Map))] for _ in range(len(visited_Map[0]))]
    visited_A_star = [[None for _ in range(len(visited_Map))] for _ in range(len(visited_Map[0]))]
    visited_A_star[x][y] = 1

    direction_list =[]
    next_x = x
    next_y = y
    for i in range(len(visited_Map)):
        for j in range(len(visited_Map[0])):
            Heuristic[i][j] = abs(i - Battery_position[0]) + abs(j - Battery_position[1]) + abs(x - i) + abs(y - j)
            if (visited_Map[i][j] == 'obstacle'):
                Heuristic[i][j] = 1e10
            elif (visited_Map[i][j] == 'empty'):
                Heuristic[i][j] = 4*Heuristic[i][j]*Heuristic[i][j]
    while(abs(x - Battery_position[0]) + abs(y - Battery_position[1])!= 0):
        visited_A_star[x][y] = 1
        Min_Cost = 1e9
        direction = ""
        if (x-1>=0 ):
            if (Heuristic[x-1][y]<Min_Cost and visited_A_star[x-1][y]!=1):
                Min_Cost = Heuristic[x-1][y]
                direction = "up"
                next_x = x-1
                next_y = y

        if (y-1>=0 ):
            if (Heuristic[x][y-1]<Min_Cost and visited_A_star[x][y-1]!=1):
                Min_Cost = Heuristic[x][y-1]
                direction = "left"
                next_x = x
                next_y = y-1
            elif (direction == "up" and Heuristic[x][y-1]==Min_Cost and visited_A_star[x][y-1]!=1):
                if (abs(x - Battery_position[0])+abs(y-1-Battery_position[1]) < abs(x-1 - Battery_position[0])+abs(y-Battery_position[1])):
                    Min_Cost = Heuristic[x][y-1]
                    direction = "left"
                    next_x = x
                    next_y = y-1

        if (x+1<len(visited_Map)):
            if (Heuristic[x+1][y]<Min_Cost  and visited_A_star[x+1][y]!=1):
                Min_Cost = Heuristic[x+1][y]
                direction = "down"
                next_x = x+1
                next_y = y
            elif (direction == "up" and Heuristic[x+1][y]==Min_Cost and visited_A_star[x+1][y]!=1):
                if (abs(x+1 - Battery_position[0])+abs(y-Battery_position[1]) < abs(x-1 - Battery_position[0])+abs(y-Battery_position[1])):
                    Min_Cost = Heuristic[x+1][y]
                    direction = "down"
                    next_x = x+1
                    next_y = y
            elif (direction == "left" and Heuristic[x+1][y]==Min_Cost and visited_A_star[x+1][y]!=1):
                if (abs(x+1 - Battery_position[0])+abs(y-Battery_position[1]) < abs(x - Battery_position[0])+abs(y-1-Battery_position[1])):
                    Min_Cost = Heuristic[x+1][y]
                    direction = "down"
                    next_x = x+1
                    next_y = y

        if (y+1<len(visited_Map[0])):
            if (Heuristic[x][y+1]<Min_Cost  and visited_A_star[x][y+1]!=1):
                Min_Cost = Heuristic[x][y+1]
                direction = "right"
                next_x = x
                next_y = y+1

            elif (direction == "up" and Heuristic[x][y+1]==Min_Cost and visited_A_star[x][y+1]!=1):
                if (abs(x - Battery_position[0])+abs(y+1-Battery_position[1]) < abs(x-1 - Battery_position[0])+abs(y-Battery_position[1])):
                    Min_Cost = Heuristic[x][y+1]
                    direction = "right"
                    next_x = x
                    next_y = y+1
            elif (direction == "left" and Heuristic[x][y+1]==Min_Cost and visited_A_star[x][y+1]!=1):
                if (abs(x - Battery_position[0])+abs(y+1-Battery_position[1]) < abs(x - Battery_position[0])+abs(y-1-Battery_position[1])):
                    Min_Cost = Heuristic[x][y+1]
                    direction = "right"
                    next_x = x
                    next_y = y+1
            elif (direction == "down" and Heuristic[x][y+1]==Min_Cost and visited_A_star[x][y+1]!=1):
                if (abs(x - Battery_position[0])+abs(y+1-Battery_position[1]) < abs(x+1 - Battery_position[0])+abs(y-Battery_position[1])):
                    Min_Cost = Heuristic[x][y+1]
                    direction = "right"
                    next_x = x
                    next_y = y+1

        direction_list.append(direction)
        x = next_x
        y = next_y
    move(direction_list)


def get_bot_pos(Map):
    bot_position = []
    for i in range(len(Map)):
        for j in range(len(Map[0])):
            if Map[i][j] == 'robot':
                bot_position = [i,j]
    return bot_position

Map = mapping()


Battery_position = []
for i in range(len(Map)):
    for j in range(len(Map[0])):
        if Map[i][j] == 'Battery':
            Battery_position = [i,j]

visited_Map = [[None for _ in range(len(Map))] for _ in range(len(Map[0]))]
bot_position = get_bot_pos(Map)
visited_Map[bot_position[0]][bot_position[1]] = 'robot'
visited_Map[Battery_position[0]][Battery_position[1]] = 'Battery'

steps = 0

#move(['right','down','right','left'])
A_star(visited_Map)



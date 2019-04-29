## compute_input.py
import sys
import json

#Read data from stdin
def read_in(path):
    data_string = ""
    with open(path) as file:  
        for raw_text in file:
            lines = [x for x in raw_text.split(',') if x != '']
        for line in lines:
            data_string += line +','
    return data_string

def main():

    #get our data as an array from read_in()
    data_string = read_in(sys.argv[1])
    # data = read_in("../input/schedule.json")

    #return the sum to the output stream
    # data_json = json.dumps(data)
    # with open('./schedule.json', 'w') as outfile:  
    # 	json.dump(data_json, outfile)
    print(data_string)

#start process
if __name__ == '__main__':
    main()
    # print('file saved')
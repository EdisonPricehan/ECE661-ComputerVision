import csv

if __name__ == '__main__':
    a = [[1,2,3],
         [4,5,6,7,8]]
    with open('test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(a)

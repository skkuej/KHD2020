import requests
import argparse


def submit(result):
    task_no = 2
    url = "http://mnc.aifactory.space/sub"
    user_id = 'line7220'
    pwd = 'mipl85765'

    basepath = '/USER/USER_WORKSPACE/hebin/cancer/'
    result_file = open(f'{basepath}results/{result}.csv','rb')

    upload = {'file': result_file}

    data = {
        "taskId": f'M00000{task_no}',
        "id": user_id,
        "pwd": pwd,
        "modelNm": f'hb_{result}'
    }

    res = requests.post(url, files=upload, data=data)

    print(res)
    #print(res.text)
    print(res.text.encode('utf8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument('--user_id', type=str, required=True)
    parser.add_argument('--pwd', type=str, required=True)
    parser.add_argument('-n', dest='task_no', type=str, required=True)  # task_no
    parser.add_argument('-m', dest='modelnm', type=str, required=True)  # modelnm
    '''
    parser.add_argument(dest='result', type=str)  # result
    args = parser.parse_args()

    submit(result=args.result)

    # submit(
    #     task_no=1,
    #     user_id='mncadmin', # ID
    #     pwd='mnc2021!@', # password
    #     modelnm='my_first_model',
    #     result='./20210215095722632675_5ioe.csv'
    #     )

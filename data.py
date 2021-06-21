import os
import ast
from os.path import join, exists
# train_data  = [('account number is the most valuable no in the banking domain', 'account number'),
# ('the most valuable number in the insurance domain is the account number', 'account number'),
# ('every insurance individual has an account number', 'account number'),
# ('I have my own account number which is important to find teh insurance', 'account number'),
# ('he has account number safe and secure', 'account number'),
# ('account number is safe and secure', 'account number'),
# ('rebecca has her account number read somewhere which is insecure', 'account number'),
# ('She and her account number doesnt match with any of them', 'account number'),
# ('account numbers are so secure and safe', 'account number'),
# ('give me your account number for verification', 'account number'),
# ('uday your account number is 1233456', 'account number'),
# ('bill date can be an important date in insurance', 'bill date'),
# ('date in insurance is always called as bill date', 'bill date'),
# ('the most important date in insurance is always bill date', 'bill date'),
# ('I am aware of bill date in insurance docs', 'bill date'),
# ('HI I am uday and I have a bill date which is the actual date', 'bill date'),
# ('I have a bill date in my insurance docs', 'bill date'),
# ('bill date is always mandatory date', 'bill date'),
# ('Sheeba bill date is due for pay', 'bill date')]

# with open('train_data', 'w') as f:
#     f.write(str(train_data))


class PreprocessData:
    label = {'account number': 'Account_Number', 'bill date': 'Bill_Date'}

    def __init__(self, data):

        self.data = data

    def process_data(self):
        preprocessed_data = []
        print(self.data)
        for i in self.data:

            start_word_index = i[0].find(i[1])
            end_word_index = start_word_index+len(i[1])
            preprocessed_data.append((i[0], {'entities': [(start_word_index, end_word_index,
                                                           PreprocessData.label[i[1]])]}))
        print(f'preprocessed_data: {preprocessed_data}')
        return preprocessed_data


# if __name__ == "__main__":
#
#     script_dir = os.path.dirname(os.path.realpath(__file__))  # <-- absolute dir the script is in
#     rel_path = "data.txt"
#     abs_file_path = os.path.join(script_dir, rel_path)
#
#     train_data = []
#     with open('train_data', 'r') as f:
#         ds_string = ast.literal_eval(f.read())
#         x = (PreprocessData(ds_string))
#         print(x.process_data())
#
#

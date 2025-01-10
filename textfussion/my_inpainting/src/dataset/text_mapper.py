import ipdb
import torch
import string
import numpy as np

voc = list(string.printable[:-6])

EOS = "EOS"
PADDING = "PADDING"
UNKNOWN = "UNKNOWN"

voc.append("EOS")
voc.append("PADDING")
voc.append("UNKNOWN")


# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
# , 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X
# ', 'Y', 'Z', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
# '`', '{', '|', '}', '~']

class TextMapper:

    def __init__(self, max_length=25):
        self.max_len = max_length
        self.char2id_dict = {}
        self.voc = voc
        for i, item in enumerate(voc):
            self.char2id_dict[item] = i

    def get_text_tokens(self, examples):
        stringlist = [example["ori_text"] for example in examples]

        # maxlength = max(len(s) for s in stringlist) + 2
        # maxlength = maxlength if maxlength < self.max_len else self.max_len
        input_ids = []

        for string in stringlist:
            if len(string) > self.max_len - 2:
                string = string[:self.max_len - 2]
            string_ids = []
            for char in string:
                if char in self.voc:
                    string_ids.append(self.char2id_dict[char])
                else:
                    string_ids.append(self.char2id_dict[UNKNOWN])
            string_ids = np.array(string_ids)
            string_ids = np.insert(string_ids, 0, self.char2id_dict[EOS])
            string_ids = np.insert(string_ids, len(string_ids), self.char2id_dict[EOS])
            string_ids = np.pad(string_ids, (0, self.max_len - len(string_ids)), 'constant',
                                constant_values=(-1, self.char2id_dict[PADDING]))
            input_ids.append(string_ids)

        input_ids = np.array(input_ids)
        # tag_array = np.array([self.char2id_dict[EOS]] * len(input_ids))
        # input_ids = np.column_stack((tag_array, input_ids, tag_array))
        input_ids = torch.from_numpy(input_ids)

        return input_ids

    def get_test_text_tokens(self, test_string):

        if isinstance(test_string, list):
            res_string = []
            for string_item in test_string:
                if len(string_item) > self.max_len - 2:
                    string_item = string_item[:self.max_len - 2]
                string_ids = []
                for char in string_item:
                    if char in self.voc:
                        string_ids.append(self.char2id_dict[char])
                    else:
                        string_ids.append(self.char2id_dict[UNKNOWN])
                string_ids = np.array(string_ids)
                string_ids = np.insert(string_ids, 0, self.char2id_dict[EOS])
                string_ids = np.insert(string_ids, len(string_ids), self.char2id_dict[EOS])
                string_ids = np.pad(string_ids, (0, self.max_len - len(string_ids)), 'constant',
                                    constant_values=(-1, self.char2id_dict[PADDING]))

                string_ids = torch.from_numpy(string_ids)
                res_string.append(string_ids)
            res_string = torch.stack(res_string)

            return res_string
        else:
            if len(test_string) > self.max_len - 2:
                test_string = test_string[:self.max_len - 2]
            string_ids = []
            for char in test_string:
                if char in self.voc:
                    string_ids.append(self.char2id_dict[char])
                else:
                    string_ids.append(self.char2id_dict[UNKNOWN])
            string_ids = np.array(string_ids)
            string_ids = np.insert(string_ids, 0, self.char2id_dict[EOS])
            string_ids = np.insert(string_ids, len(string_ids), self.char2id_dict[EOS])
            string_ids = np.pad(string_ids, (0, self.max_len - len(string_ids)), 'constant',
                                constant_values=(-1, self.char2id_dict[PADDING]))

            string_ids = torch.from_numpy(string_ids).unsqueeze(0)

            return string_ids

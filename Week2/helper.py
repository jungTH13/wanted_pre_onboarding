import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split

def set_device():
    #torch에서 GPU가 사용가능할 경우 device 설정을 GPU로 설정(없을 경우 CPU)하여 연산
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        
    return device
    
def custom_collate_fn(batch):
    """
    - batch: list of tuples (input_data(string), target_data(int))
    
    한 배치 내 문장들을 tokenizing 한 후 텐서로 변환함. 
    이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용
    토큰 개수는 배치 내 가장 긴 문장으로 해야함.
    또한 최대 길이를 넘는 문장은 최대 길이 이후의 토큰을 제거하도록 해야 함
    토크나이즈된 결과 값은 텐서 형태로 반환하도록 해야 함
    
    한 배치 내 레이블(target)은 텐서화 함.
    
    (input, target) 튜플 형태를 반환.
    """
    #토크나이저를 불러와 사용
    tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")
    
    #패키징되어 있는 데이터를 분리
    input_list, target_list = [a[0] for a in batch],[a[1] for a in batch]
    
    #분리한 문자열 데이터를 토큰화 진행(토큰화된 문장중 가장 긴 문장을 기준으로 패딩)
    tensorized_input = tokenizer_bert(input_list,
                                        add_special_tokens=True, #[CLS],[SEP]의 스페셜토큰 삽입
                                        return_tensors='pt', # 반환 타입 설정(tensor)
                                        padding='longest' #패딩 방식 설정
                                        )
    
    tensorized_label = torch.tensor(target_list)#target 리스트를 tensor로 변경
    
    return (tensorized_input, tensorized_label)


class CustomDataset(Dataset):
    """
    - input_data: list of string
    - target_data: list of int
    """

    def __init__(self, input_data:list, target_data:list) -> None:
        self.X = input_data #데이터의 입력 문자열 리스트를 저장
        self.Y = target_data #타겟 정보의 리스트를 저장

    def __len__(self):
        return len(self.Y) #학습 데이터의 길이를 반환

    def __getitem__(self, index):
        
        return self.X[index], self.Y[index] #호출한 인덱스의 정보를 반환
  
  
class CustomClassifier(nn.Module):

    def __init__(self, hidden_size: int, n_label: int):
        super(CustomClassifier, self).__init__() #부모 클래스의 생성자 초기화

        self.bert = BertModel.from_pretrained("klue/bert-base") #한국어 bert모델을 생성

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, linear_layer_hidden_size), #bert 모델의 마지막 layer의 [CLS] hidden state의 신경망 연산
            nn.ReLU(), #활성화 함수 설정(linear_layer_hidden_size 수 만큼 출력)
            nn.Dropout(dropout_rate), #앞 신경망의 노드중 dropout_rate만큼의 노드 연산을 중지(신경망 일부 노드의 과접합을 방지)
            nn.Linear(linear_layer_hidden_size,n_label) #classification을 위한 마지막 layer
            )
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )#bert 모델을 통한 hidden states 생성

        cls_token_last_hidden_states = outputs['pooler_output'] #마지막 layer의 [CLS] 토큰 정보 분리

        logits = self.classifier(cls_token_last_hidden_states) #분류 연산 실행

        return logits #실행값 반환
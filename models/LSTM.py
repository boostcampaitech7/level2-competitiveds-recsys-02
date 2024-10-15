import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# LSTM 모델 정의 클래스
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        """
        LSTM 모델 초기화 함수
          param 
            input_size: 입력 데이터의 feature 개수
            hidden_size: LSTM의 은닉층 크기
            output_size: 예측할 출력 크기 (타겟 변수의 차원)
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # LSTM 레이어
        self.fc = nn.Linear(hidden_size, output_size)  # 최종 예측을 위한 완전 연결층
        self.initialize_weights()  # 가중치 초기화

    def forward(self, x):
        """
        모델의 순전파 함수
        param 
          x: 입력 데이터
        return
          : 예측 값
        """
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])  # 마지막 은닉 상태를 사용해 예측값 생성
        return out.view(-1, 1)  # 출력 shape을 (batch_size, 1)로 변환

    def initialize_weights(self):
        """
        LSTM 및 Linear 레이어의 가중치 초기화 함수
        """
        for layer in self.children():
            if isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

# LSTM 모델을 다루는 클래스
class UsingLSTM:
    def __init__(self, df: pd.DataFrame):
        """
        UsingLSTM 클래스 초기화 함수
          param 
            df: 입력 데이터프레임
        """
        self.df = df

    def create_sequences(self, data, time_steps):
        """
        시계열 데이터를 학습할 수 있도록 시퀀스를 생성하는 함수
          param 
            data: 시퀀스를 만들 데이터
            time_steps: 각 시퀀스의 길이
          return
            : 시퀀스로 변환된 numpy 배열
        """
        sequences = []
        for i in range(len(data) - time_steps + 1):
            seq = data[i:i + time_steps]  # time_steps 길이만큼의 시퀀스 생성
            sequences.append(seq)
        return np.array(sequences)

    def df_to_tensor(self, pred_col: str = 'deposit_by_area', data_type='val', time_steps: int = 10):
        """
        데이터프레임을 학습에 사용될 텐서로 변환하는 함수
          param 
            pred_col: 예측할 타겟 열
            data_type: 'val' 또는 'test'로 데이터셋을 구분
            time_steps: 시퀀스의 길이
          return
            : 학습용 및 테스트용 입력/타겟 텐서
        """
        # 데이터 분할 (학습용, 검증/테스트용)
        if data_type == 'val':
            train = self.df[self.df['contract_year_month'] < 202307]
            test = self.df[(self.df['contract_year_month'] >= 202307) & (self.df['contract_year_month'] < 202401)]
        elif data_type == 'test':
            train = self.df[self.df['contract_year_month'] < 202001]
            test = self.df[self.df['contract_year_month'] >= 202001]
        
        # 예측에 사용할 입력 변수와 타겟 변수 설정
        X_train = train.drop(['deposit_by_area', 'deposit'], axis=1)
        y_train = train[pred_col]
        X_test = test.drop(['deposit_by_area', 'deposit'], axis=1) 
        y_test = test[pred_col]

        # 입력 변수 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 시퀀스 생성
        X_train_seq = self.create_sequences(X_train_scaled, time_steps)
        y_train_seq = self.create_sequences(y_train.values.reshape(-1, 1), time_steps)

        # y_train_tensor를 마지막 타임스텝의 값만 선택하여 2D 텐서로 변환
        y_train_tensor = torch.FloatTensor(y_train_seq[:, -1])  # 마지막 타임스텝의 값만 선택

        # 테스트 데이터에 대한 시퀀스 생성
        X_test_seq = self.create_sequences(X_test_scaled, time_steps)
        y_test_seq = self.create_sequences(y_test.values.reshape(-1, 1), time_steps)

        # y_test_tensor도 마지막 타임스텝의 값만 선택
        y_test_tensor = torch.FloatTensor(y_test_seq[:, -1])

        # 텐서로 변환
        self.X_train_tensor = torch.FloatTensor(X_train_seq)
        self.y_train_tensor = y_train_tensor  
        self.X_test_tensor = torch.FloatTensor(X_test_seq)
        self.y_test_tensor = y_test_tensor  

        return train, test, self.X_train_tensor, self.y_train_tensor, self.X_test_tensor, self.y_test_tensor
   
    def apply_LSTM(self, hidden_size: int = 64, output_size: int = 1, learning_rate: float = 0.001, batch_size: int = 64, num_epochs: int = 100):
      """
        LSTM 모델 학습 및 검증 함수
          param 
            hidden_size: LSTM의 은닉층 크기
            output_size: 타겟 변수의 크기
            learning_rate: 학습률
            batch_size: 미니 배치 크기
            num_epochs: 에폭 수 (반복 학습 횟수)
          return
            : 검증 데이터에 대한 예측 값
      """
      # 입력 데이터의 feature 개수
      input_size = self.X_train_tensor.shape[2]  

      # 모델 초기화 및 가중치 초기화
      model = LSTMModel(input_size, hidden_size, output_size).to('cuda')

      # 손실 함수 및 옵티마이저 설정
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      # 학습 데이터 로더 생성
      train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor.view(-1, 1))  # y_train_tensor를 2D로 변환
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

      # 학습 과정
      for epoch in range(num_epochs):
          model.train()
          for inputs, targets in train_loader:
              inputs, targets = inputs.to('cuda'), targets.view(-1, 1).to('cuda')
              outputs = model(inputs)
              loss = criterion(outputs, targets)

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

          if (epoch + 1) % 10 == 0:
              print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

      # 검증/테스트 데이터에 대한 예측
      model.eval()
      with torch.no_grad():
          test_outputs = []
          input_seq = self.X_test_tensor[0].unsqueeze(0).to('cuda')  # 첫 번째 시퀀스를 입력으로 사용

          for i in range(len(self.X_test_tensor)):
              output = model(input_seq)  # 모델로 예측
              test_outputs.append(output.cpu().numpy())  # CPU로 이동하여 결과 저장
              print("input_seq shape:", input_seq.shape)
              print("output shape:", output.shape)

              # 다음 입력으로 예측값을 사용하여 시퀀스 업데이트
              if i + 1 < len(self.X_test_tensor):
                  output = output.unsqueeze(1)  # output의 shape을 (1, 1)로 변경
                  input_seq = torch.cat((input_seq[:, 1:, :], output.expand(1, 1, 30)), dim=1)  # 예측값을 맞는 shape으로 확장

      test_outputs_cpu = np.array(test_outputs).squeeze()

      # 예측 결과를 원래 스케일로 되돌리기
      # test_outputs_cpu = scaler.inverse_transform(test_outputs_cpu)

      return test_outputs_cpu
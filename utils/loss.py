import torch
import torch.nn.functional as F

def mld_loss(student_logits, teacher_logits, temperature=1.0):

    student_logits_t = student_logits / temperature
    teacher_logits_t = teacher_logits / temperature
    
    loss = 0

    for i in range(student_logits.shape[1]):
        # [p_yes, p_no] 형태의 확률 분포 생성
        # log_softmax는 입력으로 log-probabilities를 기대하므로 logits를 그대로 사용
        student_dist = F.log_softmax(torch.stack([student_logits_t[:, i], -student_logits_t[:, i]], dim=1), dim=1)
        teacher_dist = F.softmax(torch.stack([teacher_logits_t[:, i], -teacher_logits_t[:, i]], dim=1), dim=1)
        
        # KL Divergence 계산
        # reduction='batchmean'은 배치 전체의 평균 손실을 계산
        loss += F.kl_div(student_dist, teacher_dist, reduction='batchmean', log_target=False)
        
    return loss
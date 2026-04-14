def _train_item(self, u):
    seq    = self.user_train[u]
    tokens, pos_labels = [], []

    for item in seq:
        if random.random() < self.mask_prob:
            # ✅ 원본 레포: MASK만 사용 (80/10/10 없음)
            tokens.append(self.mask_token)
            pos_labels.append(item)
        else:
            tokens.append(item)
            pos_labels.append(0)

    tokens     = tokens[-self.maxlen:]
    pos_labels = pos_labels[-self.maxlen:]

    pad_len    = self.maxlen - len(tokens)
    tokens     = [0] * pad_len + tokens
    pos_labels = [0] * pad_len + pos_labels

    return (
        torch.LongTensor(tokens),
        torch.LongTensor(pos_labels),
        torch.LongTensor([0] * self.maxlen)   # dummy neg
    )

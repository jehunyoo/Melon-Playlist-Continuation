import numpy as np
import pandas as pd
from khaiii import KhaiiiApi


class Title_to_tag:

    def __init__(self, train_path, val_path):

        train = pd.read_json(train_path, encoding='utf-8')
        val = pd.read_json(val_path, encoding='utf-8')

        self.train_tags = train["tags"].copy()
        self.val_id = val["id"].copy()
        self.val_title = self.re_sub(val["plylst_title"].copy())
        self.val_songs = val["songs"].copy()
        self.val_tags = val["tags"].copy()
        self.val_updt_date = val["updt_date"].copy()

    def re_sub(self, series):
        series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
        series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
        series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
        series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
        return series

    def flatten(self, list_of_list):
        flatten = [j for i in list_of_list for j in i]
        return flatten

    def get_token(self, title, tokenizer):
        if len(title) == 0 or title == ' ':  # 제목이 공백인 경우 tokenizer에러 발생
            return []

        result = tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
        return result

    def make_tag(self, val_title):
      candidate_tag = val_title.split()
      original_tag = []
      new_tag = []
      for i in candidate_tag:
        if i in self.train_tags_set:
          original_tag.append(i)
        else:
          new_tag.append(i)
      del candidate_tag

      tokenizer = KhaiiiApi()
      token_tag = [self.get_token(x, tokenizer) for x in new_tag]
      origin_khai_tag = self.flatten(token_tag)
      origin_khai_tag = [origin_khai_tag[i] for i in range(len(origin_khai_tag)) if
                         origin_khai_tag[i][0] in self.train_tags_set]
      using_pos = ['NNG', 'SL', 'NNP', 'MAG', 'SN', 'XR']  # 일반 명사, 외국어, 고유 명사, 일반 부사, 숫자, 형용사??(잔잔)
      origin_khai_tag = [origin_khai_tag[i][0] for i in range(len(origin_khai_tag)) if
                         origin_khai_tag[i][1] in using_pos]

      born_tag = list(set(origin_khai_tag + original_tag))

      return born_tag

    def change(self):
        val = []
        self.train_tags_set = set(self.flatten(self.train_tags))
        for uth in range(0, len(self.val_id)):

            if len(self.val_tags[uth]) != 0 or len(self.val_songs[uth]) != 0:
                val.append({
                    "id": int(self.val_id[uth]),
                    "songs": self.val_songs[uth],
                    "tags": self.val_tags[uth],
                    "updt_date": self.val_updt_date[uth]
                })
                continue

            val_title = self.val_title[uth]
            born_tag = self.make_tag(val_title)

            if len(born_tag) == 0:
              born_tag = self.make_tag(val_title.lower())


            if len(born_tag) == 0:
              for i in range(len(val_title)-4):
                if val_title[i:i+5] == "크리스마스": # khaiii ...
                  born_tag = [val_title[i:i+5]]

            val.append({
                "id": int(self.val_id[uth]),
                "songs": [],
                "tags": born_tag,
                "updt_date": self.val_updt_date[uth]
            })
        return pd.DataFrame(val)

if __name__ == "__main__":
    pass
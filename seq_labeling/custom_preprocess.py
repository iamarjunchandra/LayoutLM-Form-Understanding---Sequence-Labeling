import os
import numpy as np
from PIL import Image
import pytesseract
from transformers import AutoTokenizer


class custom_img_annotation_:
    def write_annoteFile(self,url):
        self.file=url
        self.file_name=os.path.basename(self.file)
        self.image=Image.open(r'{}'.format(self.file))
        self.width, self.length=self.image.size
        #OCR
        self.ocr_df = pytesseract.image_to_data(self.image, output_type='data.frame')     
        self.ocr_df = self.ocr_df.dropna() 
        self.ocr_df['text']=self.ocr_df.text.apply(lambda x:x.strip())
        self.ocr_df['text']=self.ocr_df.text.apply(lambda x:np.nan if x=='' else x)
        self.ocr_df.dropna(inplace=True)
        self.ocr_df.reset_index(inplace = True, drop = True)
        self.annotation_list=[]
        for id,x in self.ocr_df.iterrows():
            self.annotation_list.append({
                    "id":id,
                    "text":x.text,
                    "box":[x.left,x.top,x.width+x.left,
                            x.height+x.top]
            })
        return self.ocr_df
            
    def bbox_string(self, box, width, length):
        return (
            str(int(1000 * (box[0] / width)))
            + " "
            + str(int(1000 * (box[1] / length)))
            + " "
            + str(int(1000 * (box[2] / width)))
            + " "
            + str(int(1000 * (box[3] / length)))
        )

    def actual_bbox_string(self, box, width, length):
        return (
            str(box[0])
            + " "
            + str(box[1])
            + " "
            + str(box[2])
            + " "
            + str(box[3])
            + "\t"
            + str(width)
            + " "
            + str(length)
        )        
        
    def convert(self,data_split="test",output_dir="data",
            model_name_or_path="bert-base-uncased",max_len=510):
        with open(
            os.path.join(output_dir, data_split + ".txt.tmp"),
            "w",
            encoding="utf8",
        ) as fw, open(
            os.path.join(output_dir, data_split + "_box.txt.tmp"),
            "w",
            encoding="utf8",
        ) as fbw, open(
            os.path.join(output_dir, data_split + "_image.txt.tmp"),
            "w",
            encoding="utf8",
        ) as fiw:
            for file in self.annotation_list:
                self.text=file['text']
                if self.text.strip()=='':
                    continue
                fw.write(self.text + "\tO\n")
                fbw.write(
                        self.text
                        + "\t"
                        + self.bbox_string(file['box'], self.width, self.length)
                        + "\n"
                    )
                fiw.write(
                        self.text
                        + "\t"
                        + self.actual_bbox_string(file['box'], self.width, self.length)
                        + "\t"
                        + self.file_name
                        + "\n"
                    )

                fw.write("\n")
                fbw.write("\n")
                fiw.write("\n")
                
    def seg_file(self, file_path, tokenizer, max_len):
        self.subword_len_counter = 0
        self.output_path = file_path[:-4]
        with open(file_path, "r", encoding="utf8") as f_p, open(
            self.output_path, "w", encoding="utf8"
        ) as fw_p:
            for line in f_p:
                self.line = line.rstrip()

                if not self.line:
                    fw_p.write(self.line + "\n")
                    self.subword_len_counter = 0
                    continue
                self.token = self.line.split("\t")[0]

                self.current_subwords_len = len(tokenizer.tokenize(self.token))

                # Token contains strange control characters like \x96 or \x95
                # Just filter out the complete line
                if self.current_subwords_len == 0:
                    continue

                if (self.subword_len_counter + self.current_subwords_len) > max_len:
                    fw_p.write("\n" + self.line + "\n")
                    self.subword_len_counter = self.current_subwords_len
                    continue

                self.subword_len_counter += self.current_subwords_len

                fw_p.write(self.line + "\n")
    
    def seg(self, data_split="test",output_dir="data",
        model_name_or_path="bert-base-uncased",max_len=510):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=True
        )
        self.seg_file(
            os.path.join(output_dir, data_split + ".txt.tmp"),
            self.tokenizer,
            max_len,
        )
        self.seg_file(
            os.path.join(output_dir, data_split + "_box.txt.tmp"),
            self.tokenizer,
            max_len,
        )
        self.seg_file(
            os.path.join(output_dir, data_split + "_image.txt.tmp"),
            self.tokenizer,
            max_len,
        )
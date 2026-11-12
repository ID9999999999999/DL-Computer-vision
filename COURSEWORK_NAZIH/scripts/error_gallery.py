"""Render source-authenticated failed-image examples without rerunning the model.

Requires the separately supplied official KolektorSDD2.zip, the archived
ERROR_CASES.csv and Pillow. Outputs a photographic case sheet: each panel
contains the real original test image and known classification decisions.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

EXPECTED_ARCHIVE_SHA = 'edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8'


def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    assert hash_file(args.archive)==EXPECTED_ARCHIVE_SHA, 'Official dataset ZIP changed: refuse image interpretation'
    with args.cases.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert rows and all(r['image'].startswith('test/') and r['image'].endswith('.png') for r in rows)
    # Full-size source image shown scaled uniformly for layout; no synthetic textures.
    tile_w,tile_h=350,640
    cols=3
    board=Image.new('RGB',(cols*tile_w,((len(rows)+cols-1)//cols)*tile_h),'white')
    draw=ImageDraw.Draw(board)
    font=ImageFont.load_default()
    with zipfile.ZipFile(args.archive) as zipf:
        for i,r in enumerate(rows):
            raw=zipf.read(r['image'])
            assert hashlib.sha256(raw).hexdigest()==r['image_sha256'], 'Original image SHA256 mismatch: '+r['image']
            with Image.open(io.BytesIO(raw)) as img:
                im=img.convert('RGB')
                im.thumbnail((tile_w-18,tile_h-125))
            x=(i%cols)*tile_w;y=(i//cols)*tile_h
            board.paste(im,(x+(tile_w-im.width)//2,y+72))
            for j,line in enumerate((r['image'],r['actual_label'],
                 f"CE: {r['ce_prediction']} ({r['ce_logit_margin']})",
                 f"Poly-1: {r['poly1_prediction']} ({r['poly1_logit_margin']})")):
                draw.text((x+10,y+10+14*j),line,font=font,fill='black')
            draw.rectangle((x,y,x+tile_w-1,y+tile_h-1),outline=(150,150,150),width=1)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    board.save(args.out)
    print('VERIFIED GALLERY:',args.out,'samples:',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())

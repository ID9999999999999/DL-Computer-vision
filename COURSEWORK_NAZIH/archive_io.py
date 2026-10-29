"""Read selected ZIP members with HTTP ranges, validating ZIP CRCs.

Never falls back to downloading the complete archive. Only normal fit/dev
images are selected; held-out test images and calibration images stay closed.
"""
import io,json,urllib.request,zipfile,hashlib,time,struct,zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SIZE=14955220087
REV=(ROOT/'revision.txt').read_text().strip()
URL=f'https://huggingface.co/datasets/ChengYuQi99/M2AD/resolve/{REV}/Motor.zip'

class RemoteFile(io.RawIOBase):
    def __init__(self): self.pos=0;self.transferred=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos+offset if whence==1 else SIZE+offset)
        if self.pos<0: raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else SIZE-self.pos,SIZE-self.pos)
        if n<=0:return b''
        if n>20_000_000:raise ValueError(f'Refusing excessive range: {n}')
        start,end=self.pos,self.pos+n-1
        req=urllib.request.Request(URL+f'?range_start={start}&range_end={end}',headers={'Range':f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=45) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Server did not honour exact byte range')
                    b=r.read(n+1)
                if len(b)!=n:raise ValueError('Incomplete range')
                self.pos+=n;self.transferred+=n;return b
            except Exception:
                if attempt==2:raise
                time.sleep(1)

def read_member(remote, info):
    # Local filename/extra-field sizes may differ from central-directory values.
    remote.seek(info.header_offset)
    block=remote.read(min(info.compress_size+131100,SIZE-info.header_offset))
    if block[:4]!=b'PK\x03\x04': raise ValueError('Invalid local ZIP header')
    n,e=struct.unpack_from('<HH',block,26)
    start=30+n+e
    compressed=block[start:start+info.compress_size]
    if len(compressed)!=info.compress_size:raise ValueError('Truncated member')
    if info.compress_type==zipfile.ZIP_DEFLATED: data=zlib.decompress(compressed,-15)
    elif info.compress_type==zipfile.ZIP_STORED:data=compressed
    else:raise ValueError('Unsupported compression')
    if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('CRC or size mismatch')
    return data


"""Minimal shapefile/dbf reader (no GDAL needed). Used by 01_offset_LCDB.py.
Source: chat "Pasture stack treatment comparison"."""
import struct

def read_dbf(p):
    b=open(p,'rb').read(); n=struct.unpack('<I',b[4:8])[0]; hl,rl=struct.unpack('<HH',b[8:12])
    f=[];o=32
    while b[o]!=0x0D:
        name=b[o:o+11].split(b'\0')[0].decode(); t=chr(b[o+11]); L=b[o+16]; f.append((name,t,L)); o+=32
    recs=[]
    for i in range(n):
        s=hl+i*rl; rec=b[s:s+rl]
        if rec[0:1]==b'*': continue
        pos=1; d={}
        for name,t,L in f:
            v=rec[pos:pos+L].decode('utf-8','replace').strip(); pos+=L
            if t in 'NF': v=float(v) if v else None
            d[name]=v
        recs.append(d)
    return f,recs

def read_shp(p):
    b=open(p,'rb').read(); st=struct.unpack('<i',b[32:36])[0]; bbox=struct.unpack('<4d',b[36:68])
    o=100; shapes=[]
    while o<len(b):
        L=struct.unpack('>i',b[o+4:o+8])[0]*2; c=b[o+8:o+8+L]; o+=8+L
        t=struct.unpack('<i',c[:4])[0]
        if t==0: shapes.append([]); continue
        np_,npt=struct.unpack('<ii',c[36:44]); parts=list(struct.unpack(f'<{np_}i',c[44:44+4*np_]))
        pts=struct.unpack(f'<{2*npt}d',c[44+4*np_:44+4*np_+16*npt]); pts=list(zip(pts[0::2],pts[1::2]))
        parts.append(npt); shapes.append([pts[parts[i]:parts[i+1]] for i in range(np_)])
    return st,bbox,shapes

import re, numpy as np
# ---------------- read tif (rasterio -> tifffile+imagecodecs -> pure-Python LZW) ----------------
def _lzw(data):
    out=bytearray(); tab=[bytes([i]) for i in range(256)]+[b"",b""]; pos=0; nb=9; prev=None; n=len(data)*8
    while pos+nb<=n:
        by=pos>>3; v=int.from_bytes(data[by:by+4].ljust(4,b"\0"),"big"); c=(v>>(32-(pos&7)-nb))&((1<<nb)-1); pos+=nb
        if c==256: tab=tab[:258]; nb=9; prev=None; continue
        if c==257: break
        if prev is None: e=tab[c]; out+=e; prev=e; continue
        if c<len(tab): e=tab[c]; tab.append(prev+e[:1])
        else: e=prev+prev[:1]; tab.append(e)
        out+=e; prev=e
        if len(tab)+1>=(1<<nb) and nb<12: nb+=1
    return bytes(out)
def read_stack(path):
    try:
        import rasterio
        with rasterio.open(path) as r: return np.moveaxis(r.read().astype("float32"),0,-1), list(r.descriptions)
    except ImportError: pass
    import tifffile
    t=tifffile.TiffFile(path); p=t.pages[0]
    names=re.findall(r">([^<]+)</Item>", p.tags["GDAL_METADATA"].value)
    try: return p.asarray().astype("float32"), names
    except Exception: pass
    H,W,B=p.shape; dt=np.dtype(p.dtype).newbyteorder(t.byteorder); arr=np.zeros((H,W,B),"float32"); fh=t.filehandle
    if p.is_tiled:
        th,tw=p.tilelength,p.tilewidth; nx=-(-W//tw)
        for i,(o,c) in enumerate(zip(p.dataoffsets,p.databytecounts)):
            fh.seek(o); tile=np.frombuffer(_lzw(fh.read(c))[:th*tw*B*4],dt).reshape(th,tw,B)
            r,cc=divmod(i,nx); r*=th; cc*=tw; arr[r:r+th,cc:cc+tw]=tile[:min(th,H-r),:min(tw,W-cc)]
    else:
        rps=p.rowsperstrip
        for i,(o,c) in enumerate(zip(p.dataoffsets,p.databytecounts)):
            fh.seek(o); r=i*rps; nr=min(rps,H-r)
            arr[r:r+nr]=np.frombuffer(_lzw(fh.read(c))[:nr*W*B*4],dt).reshape(nr,W,B)
    return arr, names


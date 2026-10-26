def doa_to_str(doa):
    if doa < -0.6:
        direction = "on the left side"
    elif doa < -0.2:
        direction = "in the front-left"
    elif doa < 0.2:
        direction = "in front"
    elif doa < 0.6:
        direction = "in the front-right"
    else:
        direction = "on the right side"
    return direction

def spatialize_caption(caption, doa, with_spatial):
    # 文字列整形
    caption = caption.strip()
    caption = caption.rstrip('.')
    caption = caption.strip()
    caption = caption[0].upper() + caption[1:] if caption else caption

    # doaに基づく方向表現
    direction = doa_to_str(doa)

    if with_spatial:
        return f"{caption} {direction}."
    else:
        return f"{caption}."

def meta_to_caption(metas, with_spatial=True, sort_on_doa=False):
    lis = [
        (spatialize_caption(meta["caption"], meta["doa"], with_spatial), meta["doa"])
        for meta in metas
    ]
    lis.sort(key=lambda x:x[1])
    return " ".join([x[0] for x in lis])

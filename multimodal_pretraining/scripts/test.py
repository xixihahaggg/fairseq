from collections import defaultdict

if __name__ == '__main__':
    dic = {}
    a = "xxxxx"
    dic.setdefault(a, {})
    print(dic)
    dic[a].setdefault("hhh", {})["xxi"] = 1
    print(dic)
    dic[a].setdefault("hhh", {})["xxxxi"] = 2
    print(dic)

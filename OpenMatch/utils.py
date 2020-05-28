from argparse import Action

class DictOrStr(Action):
    def __call__(self, parser, namespace, values, option_string=None):
         if '=' in values:
             my_dict = {}
             for kv in values.split(","):
                 k,v = kv.split("=")
                 my_dict[k] = v
             setattr(namespace, self.dest, my_dict)
         else:
             setattr(namespace, self.dest, values)

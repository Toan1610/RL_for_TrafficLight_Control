from src.controller.base_controller import BaseController
from src.controller import register

@register("fixed_time")
class FixedTime(BaseController):

    def start(self):
        pass

    def action(self, t):
        # no need to action
        return 999999999

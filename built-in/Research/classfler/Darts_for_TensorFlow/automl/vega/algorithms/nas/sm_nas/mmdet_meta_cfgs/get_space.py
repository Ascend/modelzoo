"""Get search space for SM-NAS."""


class get_space(dict):
    """Class of get search space.

    :param name: name
    :type name: str
    """

    def __init__(self, name, *args, **kwargs):
        super(get_space, self).__init__(*args, **kwargs)
        self._name = name
        self.__dict__ = self

    @property
    def name(self):
        """Get name."""
        return self._name

    def _register_space(self, module_class):
        """Register class with name."""
        module_name = module_class.__name__
        if module_name in self:
            raise KeyError(
                '{} is already registered in {}'.format(
                    module_name, self.name))
        self[module_name] = module_class

    def register_space(self, cls):
        """Register and Return the class."""
        self._register_space(cls)
        return cls


backbone = get_space('backbone')
neck = get_space('neck')
roi_extractor = get_space('roi_extractor')
shared_head = get_space('shared_head')
bbox_head = get_space('bbox_head')
rpn_head = get_space('rpn_head')
detector = get_space('detector')
search_space = dict(
    backbone=backbone,
    neck=neck,
    roi_extractor=roi_extractor,
    shared_head=shared_head,
    bbox_head=bbox_head,
    rpn_head=rpn_head,
    detector=detector)

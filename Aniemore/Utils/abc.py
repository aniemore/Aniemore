from abc import abstractmethod


class MasterModel:
    model_config = None
    feature_extractor = None
    processor = None
    tokenizer = None
    model = None
    device = None

    def __init__(self):
        """
        A constructor.
        """
        pass

    def to(self, a):
        """
        Setup a torch.device

        :param a:
        :return:
        """
        pass

    def setup(self):
        """
        Setup models, tokenizer, config, processor, feature extractor & etc.

        :return:
        """
        pass

    def _predict_one(self, a):
        """
        Returns a List[dict] of prediction logits

        :param a:
        :return:
        """
        pass

    def _predict_many(self, a):
        """
        Returns a List[list[str, dict]] of prediction logits

        :param a:
        :return:
        """
        pass

    def predict(self, a):
        """
        Evokes with prediction logits and has logical algorythm about
        method@_predict_one and method@_predict_many

        :param a:
        :return:
        """
        pass

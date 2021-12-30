from tensorflow.keras import Model


def construct_model(body, head):
    """ Constructs a full model from a body
    and a (classification/regression) head. """
    name = f'{body.name}-{head.name}'
    model = Model(body.input, head.output, name=name)
    return model
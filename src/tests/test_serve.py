from serve.predict import model_fn
from serve.predict import input_fn
from serve.predict import predict_fn
import pickle


def test_serve():
    model_dir = './data/modelDir'

    model = model_fn(model_dir)
    print(model)

    # input = input_fn(pickle.dumps(u"Best movie ever"), 'text/plain')
    # print(input)

    input_data = "Best movie ever"
    predict_fn(input_data=input_data, model=model)

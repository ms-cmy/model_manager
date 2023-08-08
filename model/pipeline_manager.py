from collections import defaultdict
from collections import abc
from typing import Any, Callable, Iterable, Literal, Union
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
from itertools import repeat
import pickle

class Pipes():

    pipes: dict[int,
                dict[Literal['pipe'], Pipeline]] = {}

    def add_pipe(self, pipe: Union[Pipeline, list[Pipeline], dict[str, Pipeline]]) -> None:
        if isinstance(pipe, Pipeline):
            self.pipes[len(self.pipes)] = {'pipe': pipe}
        elif isinstance(pipe, abc.Iterable):
            return self._add_pipe_iterable(pipe)
        elif isinstance(pipe, dict):
            return self._add_pipe_dict(pipe)
        else:
            raise ValueError('Invalid type')

    def _add_pipe_iterable(self, pipes: Iterable[Pipeline]) -> None:
        for pipe in pipes:
            self._check_pipe_type(pipe)
            self.add_pipe(pipe)
        return

    def _add_pipe_dict(self, pipes: dict[str, Pipeline]) -> None:
        for name, pipe in pipes.items():
            self._check_pipe_name(name)
            self._check_pipe_type(pipe)
            self.pipes[name] = pipe
        return

    def _check_pipe_name(self, name) -> None:
        if name not in self.pipes.keys():
            raise ValueError(f'Name {name} not found')
        return

    def _check_pipe_type(self, pipe: Pipeline) -> None:
        if not isinstance(pipe, Pipeline):
            raise ValueError('Invalid type')
        return

    def yield_pipes(self) -> Iterable[tuple[str, Pipeline]]:
        for name, pipe in self.pipes.items():
            yield name, pipe['pipe']

class AbstractModelPipe(Pipes):
    def __init__(self,
                 x: pd.DataFrame,
                 y: pd.Series,
                 pipes) -> None:
        self.x = x
        self.y = y
        self.add_pipe(pipes)
        pass

class CategoricalModelPipe(AbstractModelPipe):

    def __init__(self, x: pd.DataFrame,
                 y: pd.Series,
                 pipes) -> None:

        super().__init__(x,
                         y,
                         pipes)


    def fit(self, **args) -> None:
        self.generate_x_y_train_test(**args)
        self.fit_pipes()
        self.get_cat_scores()
        return

    def generate_x_y_train_test(self, **args) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, **args)
        return

    def fit_pipes(self) -> None:
        for name, pipe in self.yield_pipes():
            try:
                pipe.fit(self.x_train, self.y_train)
            except Exception as e:
                print(f'Error in {name}')
                raise e
        return

    def get_cat_scores(self) -> None:
        functions = (f1_score, accuracy_score, precision_score, recall_score)
        for name, pipe in self.yield_pipes():
            y_pred = pipe.predict(self.x_test)
            scores = [(round(func(self.y_test, y_pred), 2), func.__name__) for func in functions]
            self.pipes[name]['scores'] = scores
        return

    def get_confusion_matrix(self, model_name: str) -> None:
        pipe = self.pipes[model_name]['pipe']
        y_pred = pipe.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        disp.plot()
        return

    def save_model(self, model_name: str) -> None:
        pipe = self.pipes[model_name]['pipe']
        with open(f'{model_name}.pkl', 'wb') as f:
            pickle.dump(pipe, f)
        return

    def save_all_models(self) -> None:
        for model_name in self.pipes.keys():
            self.save_model(model_name)
        return

    def load_model(self, model_name: str) -> None:
        with open(f'{model_name}.pkl', 'rb') as f:
            pipe = pickle.load(f)
        return pipe



# generalizar os loops nas pipes

# class to manage the pipelines

# base class for the usage of pipelines

# class for continous pipelines

# class for categorical pipelines

# problem when we have way too many groups that we want to fit



class pipemanager():
    def __init__(self,
                 pipes: dict[Pipeline],
                 x: pd.DataFrame,
                 y: pd.DataFrame,
                 problem: Literal['categorical', 'continous']='categorical') -> None:
        self.pipes = self.fix_pipes_if_not_dict(pipes) # create a custom class to manage this
        self.x = x
        self.y = y
        pass

    def fix_pipes_if_not_dict(self, pipes) -> dict[str, dict[Literal['pipe'], Pipeline]]:
        return dict({i: {'pipe': pipe} for i, pipe in enumerate(pipes)})

    def generate_x_y_train_test(self, **args):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, **args)
        return

    def fit_pipes(self):
        for name, pipe in self.pipes.items():
            p = pipe['pipe']
            try:
                p.fit(self.x_train, self.y_train)
            except Exception as e:
                print(f'Error in {name}')
                raise e
        return

    def get_cat_scores(self):
        functions = (f1_score, accuracy_score, precision_score, recall_score)
        for model_name, pipe in self.pipes.items():
            p = pipe['pipe']
            y_pred = p.predict(self.x_test)
            scores = [(round(func(self.y_test, y_pred), 2), func.__name__) for func in functions]
            self.pipes[model_name]['scores'] = scores
        return

    def get_dataframes(self): # fazer com yield
        for model_name, pipe in self.pipes.items():
            p = pipe['pipe']
            precessor: Pipeline = p.named_steps['preprocessor'] # colocar em função
            df_new = precessor.fit_transform(self.x)
        return


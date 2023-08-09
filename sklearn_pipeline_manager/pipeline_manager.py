from abc import abstractmethod
from collections import defaultdict
from collections import abc
from typing import Iterable, Literal, Union
from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

ROUND_VALUE = 3

class Pipes():

    def __init__(self) -> None:
        self.pipes: dict[int,
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
        if pipes is not None:
            self.add_pipe(pipes)
        super().__init__()
        pass

    @abstractmethod
    def fit(self, **args) -> None:
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

    def result_pipe(self, pipe_name: str) -> Pipeline:
        selected_pipe: Pipeline = clone(self.pipes[pipe_name]['pipe'])
        selected_pipe.fit(self.x, self.y)
        return selected_pipe

class CategoricalModelPipe(AbstractModelPipe):

    def __init__(self, x: pd.DataFrame,
                 y: pd.Series,
                 pipes = None) -> None:

        super().__init__(x,
                         y,
                         pipes)

    def fit(self, **args) -> None:
        self.generate_x_y_train_test(**args)
        self.fit_pipes()
        self.get_cat_scores()
        return

    def generate_x_y_train_test(self, use_all_data: bool=False, **args,) -> None:
        if use_all_data:
            self.x_train = self.x
            self.x_test = self.x
            self.y_train = self.y
            self.y_test = self.y
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, **args)
        return

    def fit_pipes(self) -> None:
        for name, pipe in self.yield_pipes():
            try:
                pipe.fit(self.x_train, self.y_train.squeeze())
            except Exception as e:
                print(f'Error in {name}')
                raise e
        return

    def get_cat_scores(self, func=None) -> None:

        if func is None:
            functions = (f1_score, accuracy_score, precision_score, recall_score)
        else:
            functions = func

        for name, pipe in self.yield_pipes():
            y_pred = pipe.predict(self.x_test)
            scores = [(round(func(self.y_test, y_pred), ROUND_VALUE), func.__name__) for func in functions]
            self.pipes[name]['scores'] = scores
        return


class PlotCategoricalModel:

    def __init__(self, class_obj: CategoricalModelPipe) -> None:
        self.confirm_type(class_obj)
        self.class_obj = class_obj
        self.pipe_and_scores = self.get_pipe_and_scores()
        pass

    def get_pipe_and_scores(self) -> pd.DataFrame:
        pipe_and_scores = defaultdict(list)
        for name, pipe in self.class_obj.pipes.items():
            pipe_and_scores[name] = pipe['scores']
        pipe_and_scores = pd.DataFrame([(k,) + item for k, v in pipe_and_scores.items() for item in v],
                                       columns=['Model', 'Score', 'Metric'])

        colors = {name: plt.get_cmap('tab10')(en) for en, name in enumerate(pipe_and_scores['Model'].unique())}
        pipe_and_scores['Color'] = pipe_and_scores['Model'].map(colors)
        return pipe_and_scores

    def confirm_type(self, class_obj: CategoricalModelPipe) -> None:
        if not isinstance(class_obj, CategoricalModelPipe):
            raise ValueError(f'class_obj must be of type CategoricalModelPip not {type(class_obj)}')
        return

    def plot_scores(self):
        unique_metrics = self.pipe_and_scores['Metric'].unique()
        model_number = len(self.pipe_and_scores['Model'].unique())
        subplots, axes = plt.subplots(nrows=len(unique_metrics), ncols=1, figsize=(model_number+0.5, 2.5*len(unique_metrics)))
        subplots.tight_layout()
        for metric, plot in zip(unique_metrics, axes.flatten()):
            plot: plt.Axes
            df: pd.DataFrame = self.pipe_and_scores[self.pipe_and_scores['Metric'] == metric]
            sns.lineplot(x='Model', y='Score', data=df,
                         hue=df['Color'], legend=False,
                         marker='o', ax=plot)
            plot.set_ylabel(metric)
            plot.set_xlabel(None)
            plot.set_xticks(df['Model'].unique().tolist())
        plot.set_xlabel('Model')

    def plot_confusion_matrix(self, model_name: str) -> None:
        self.get_confusion_matrix(model_name)
        return

    def plot_all_confusion_matrix(self):
        for model_name in self.class_obj.pipes.keys():
            self.plot_confusion_matrix(model_name)
        return

    def get_confusion_matrix(self, model_name: str) -> None:
        pipe = self.class_obj.pipes[model_name]['pipe']
        y_pred = pipe.predict(self.class_obj.x_test)
        cm = confusion_matrix(self.class_obj.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        disp.plot()
        return


# generalizar os loops nas pipes

# class to manage the pipelines

# base class for the usage of pipelines

# class for continous pipelines

# class for categorical pipelines

# problem when we have way too many groups that we want to fit

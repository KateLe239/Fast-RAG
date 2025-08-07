from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import torch
import torch.nn.functional as F
from fast_plaid import search
from pylate import indexes, models, retrieve

def get_documents(path: str):
  """чтение файла из пути и преобразование его в отрывки для индексирования и поиска.
  на данный момент реализовано наивное разбиение конкретного файла по главам.

  TODO: реализовать считывание нескольких файлов
  TODO: реализовать "умное" смысловое разделение
  а что если пользователь захочет смотреть отрывки текста конкретного размера?
  """
  with open(path, 'r') as f:
    text = f.read()
  chapters = text.split('CHAPTER')[366:] #конкретно в этом файле главы обозначаются так, первые несколько элементов не имеют смысловой нагрузки
  # sentences = text.split('.') #возможность разбить на совсем маленькие отрывки

  return chapters

def get_colbert():
  model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1", #colbert от создателей pylate
    device = 'cuda',
  )

  return model



def create_embeddings(documents: list, model):
  """формирование эмбеддингов, соответствующих формату для индекса (лист тензоров)
  если получится, запустить обучение, нужно добавить загрузку из чекпоинта
  """

  documents_embeddings = model.encode(
      documents,
      batch_size=32,
      is_query=False,
      show_progress_bar=True,
      convert_to_numpy = False, #colbert дефолтно возвращает лист numpy массивов, просим его возввращать тензорами
      convert_to_tensor = True,
  )

  return documents_embeddings



def create_index(documents_embeddings, path_for_index: str):
  """создание собственно индекса. fast-plaid позволяет сделать это очень легко
  """
  index = search.FastPlaid(index=path_for_index, device = 'cuda')

  index.create(
      documents_embeddings=documents_embeddings
  )

  return index



class YourIndex():
  def __init__(
        self,
      documents_path: str,
      index_path: str,
    ) -> None:
    self.documents = get_documents(documents_path)
    self.model = get_colbert()
    documents_embeddings = create_embeddings(self.documents, self.model)
    self.index = create_index(documents_embeddings, index_path)
    self.model_for_ans = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


  def get_ranking_docs(self, questions: list[str], k: int):
    """преобразование запросов и поиск соотвествий с помощью созданного индекса (ранжирование)
    возвращает лист листов, где подлисты соотвествуют одному из вопросов

      можно аназилировать скоры чтобы понять, насколько ответ релевантен
    """

    queries_embeddings = self.model.encode(
        questions,
        batch_size=32,
        is_query=True,
        convert_to_numpy = False,
        convert_to_tensor = True,
    )

    scores = self.index.search(
          queries_embeddings=torch.stack(queries_embeddings),
          top_k=k,
      )

    suitable_docs = [] #для удобства вытаскиваем индексы из ответа модели
    for question in scores:
        current_answers = [int(doc[0]) for doc in question]
        suitable_docs.append(current_answers)

    return suitable_docs

  def get_answers(self, questions: list[str], k: int = 5):
    """
    !!!очень сырая функция!!!
    взята случайная модель, умеющая выдавать "ответ" по отрывку информации.
    этой модели подается конкатенация найденных документов в качестве контекста


    может пользователю это вообще не надо, может он хочет просто отрывки текста?
    """
    suitable_docs = self.get_ranking_docs(questions, k)

    for i in range(len(questions)):
      context = ""
      for ind in suitable_docs[i]:
        context += self.documents[ind]

      question = questions[i]
      result = self.model_for_ans(question=question, context=context)

      print(f"Вопрос: {question}")
      print(f"Ответ: {result['answer']}")
      print(f"Уверенность: {result['score']:.2f}")



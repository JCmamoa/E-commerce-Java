# E-commerce-Java

## Recomendador de e-commerce com DJL e TensorFlow

Este projeto inclui um módulo Maven que usa a arquitetura MLP descrita no
material `Recursos Principais`: camadas densas de 128 e 64 unidades, ReLU e
saída sigmoide. O DJL gerencia os tensores, e o grafo é executado pelo
TensorFlow.

O engine TensorFlow do DJL não permite montar essa rede com `Model.setBlock()`.
Por isso, a rede é definida com as operações do TensorFlow Java, mantendo os
arrays de entrada e saída no DJL.

### Executar

Requer Java 17 ou superior e Maven:

```sh
mvn test
mvn exec:java -Dexec.mainClass=br.com.ecommerce.ml.TensorOperations
```

O backend TensorFlow CPU está configurado no `pom.xml`. A primeira execução
pode baixar as bibliotecas nativas. O dataset `basicdataset` não é necessário
para esta etapa, que implementa inferência e não treinamento.

### Integração

O modelo espera um vetor de quatro features numéricas para cada par
cliente-produto: idade do cliente, ticket médio, preço e categoria codificada.
Como o modelo Java de referência não possui idade nem histórico de ticket
médio, essas features precisam ser adicionadas ou calculadas antes de usar a
inferência na vitrine. Use a mesma normalização no treinamento e na inferência.

Os pesos atuais são determinísticos e servem apenas para demonstração; os
scores **não são probabilidades calibradas e não devem ordenar a vitrine em
produção**. É necessário treinar o modelo com histórico rotulado de impressões
e compras e usar os pesos treinados para obter recomendações úteis.

FROM python:3.6

ADD /docker_agent_B /agent

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/  .
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["agent_B.py"]

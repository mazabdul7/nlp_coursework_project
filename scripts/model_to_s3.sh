#!/bin/bash

cd /home/ec2-user/nlp_coursework_project

zip -r gpt.zip sample_data/
aws s3 cp gpt.zip s3://nlp-project/

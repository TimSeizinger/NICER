#!/bin/bash
fastbuildah bud -t ghcr.io/janpf/niaa/pytorch:latest pytorch.dockerfile
fastbuildah push ghcr.io/janpf/niaa/pytorch:latest
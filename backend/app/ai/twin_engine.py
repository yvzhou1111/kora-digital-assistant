import os
import uuid
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from sqlalchemy.orm import Session
import logging
import json

# 导入Second-Me依赖
from lpm_kernel.L1.l1_generator import L1Generator
from lpm_kernel.L1.bio import Bio, Note, Cluster, ShadeInfo, Todo, Chat
from lpm_kernel.configs.config import Config

from app.models.digital_twin import DigitalTwin
from app.models.twin_data_source import TwinDataSource, TwinChunk
from app.models.twin_cluster import TwinCluster, TwinShade
from app.models.conversation import Conversation, Message
from app.core.config import settings
from app.ai.doubao_client import doubao_client

# 设置日志
logger = logging.getLogger(__name__)


class TwinEngine:
    """
    数字分身引擎类，整合Second-Me的L1Generator功能
    """
    
    def __init__(self, db: Session):
        """
        初始化数字分身引擎
        
        参数:
        - db: 数据库会话
        """
        self.db = db
        self.chroma_client = None
        self.embedding_model = settings.EMBEDDING_MODEL
        self.llm_model = settings.LLM_MODEL
        self.preferred_language = settings.PREFER_LANGUAGE
        
        # 初始化Second-Me组件
        try:
            self.l1_generator = L1Generator()
            self.l1_generator.preferred_language = self.preferred_language
            self.l1_generator.bio_model_params = {
                "temperature": settings.L1_GENERATOR_TEMPERATURE,
                "max_tokens": settings.L1_GENERATOR_MAX_TOKENS,
                "top_p": settings.L1_GENERATOR_TOP_P,
                "frequency_penalty": settings.L1_GENERATOR_FREQUENCY_PENALTY,
                "seed": settings.L1_GENERATOR_SEED,
                "presence_penalty": settings.L1_GENERATOR_PRESENCE_PENALTY,
                "timeout": settings.L1_GENERATOR_TIMEOUT,
            }
            logger.info("L1Generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize L1Generator: {str(e)}")
            self.l1_generator = None
        
        # 初始化向量数据库客户端
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
            logger.info("ChromaDB initialized successfully")
        except ImportError:
            logger.warning("ChromaDB not installed, vector search will not be available")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
    
    def generate_twin_response(self, twin_id: uuid.UUID, query: str, conversation_id: uuid.UUID) -> str:
        """
        生成数字分身回复
        
        参数:
        - twin_id: 数字分身ID
        - query: 用户查询
        - conversation_id: 对话ID
        
        返回:
        - 生成的回复文本
        """
        # 获取数字分身信息
        twin = self.db.query(DigitalTwin).filter(DigitalTwin.id == twin_id).first()
        if not twin:
            return "数字分身不存在"
        
        # 获取对话历史
        conversation_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()
        
        # 构建对话历史
        conversation_history = []
        for msg in conversation_messages:
            role = "assistant" if msg.is_from_twin else "user"
            conversation_history.append({"role": role, "content": msg.content})
        
        # 添加当前查询
        conversation_history.append({"role": "user", "content": query})
        
        # 执行向量搜索，获取相关上下文
        contexts = self._search_relevant_contexts(twin_id, query, limit=5)
        
        # 构建系统提示
        system_prompt = self._build_system_prompt(twin, contexts)
        
        # 使用LLM生成回复
        response = self._generate_llm_response(system_prompt, conversation_history)
        
        return response
    
    def _build_system_prompt(self, twin: DigitalTwin, contexts: List[Dict[str, Any]]) -> str:
        """
        构建系统提示
        
        参数:
        - twin: 数字分身对象
        - contexts: 相关上下文列表
        
        返回:
        - 系统提示文本
        """
        # 基本身份信息
        prompt = f"你是{twin.name}的数字分身。"
        
        # 添加生物特征信息
        if twin.bio_summary:
            prompt += f"\n\n关于你自己的概述:\n{twin.bio_summary}"
        
        # 添加个性特征
        if twin.personality_traits:
            traits = ", ".join([f"{k}: {v}" for k, v in twin.personality_traits.items()])
            prompt += f"\n\n你的个性特征: {traits}"
        
        # 添加交流风格
        prompt += f"\n\n你的交流风格是: {twin.communication_style}"
        
        # 添加相关上下文
        if contexts:
            prompt += "\n\n以下是与当前问题相关的信息，请参考这些信息来回答:"
            for i, ctx in enumerate(contexts):
                prompt += f"\n\n文档 {i+1}:\n{ctx['content']}"
        
        # 添加回复指导
        prompt += "\n\n请以第一人称回答问题，就像你真的是这个人一样。如果你不知道答案，请诚实地说出来，不要编造信息。"
        
        return prompt
    
    def _search_relevant_contexts(self, twin_id: uuid.UUID, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        搜索与查询相关的上下文
        
        参数:
        - twin_id: 数字分身ID
        - query: 用户查询
        - limit: 返回结果数量
        
        返回:
        - 相关上下文列表
        """
        if not self.chroma_client:
            return []
        
        try:
            # 获取数字分身的向量集合
            twin = self.db.query(DigitalTwin).filter(DigitalTwin.id == twin_id).first()
            if not twin or not twin.vector_collection_name:
                return []
            
            collection = self.chroma_client.get_collection(name=twin.vector_collection_name)
            
            # 执行向量搜索
            results = collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            # 处理结果
            contexts = []
            if results and "documents" in results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    contexts.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {},
                        "id": results["ids"][0][i] if "ids" in results and results["ids"] else None,
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    })
            
            return contexts
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []
    
    def _generate_llm_response(self, system_prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        使用LLM生成回复
        
        参数:
        - system_prompt: 系统提示
        - conversation_history: 对话历史
        
        返回:
        - 生成的回复文本
        """
        try:
            # 构建完整消息
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            
            # 使用豆包客户端生成回复
            response = doubao_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 处理响应
            content = doubao_client.process_response(response)
            if content:
                return content
            
            # 如果无法使用豆包API，返回模拟回复
            return self._mock_response(conversation_history[-1]["content"])
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"抱歉，生成回复时出现了错误: {str(e)}"
    
    def _mock_response(self, query: str) -> str:
        """
        生成模拟回复（当无法使用真实LLM时）
        
        参数:
        - query: 用户查询
        
        返回:
        - 模拟回复文本
        """
        if "介绍" in query or "你是谁" in query:
            return "我是你的数字分身。我可以帮助回答关于你的问题，代表你与他人交流。"
        elif "经验" in query or "技能" in query:
            return "根据我的知识库，我了解到你具有多种技能和经验。你想了解哪方面的具体信息？"
        elif "项目" in query:
            return "我参与过多个项目。需要我详细介绍某个特定项目吗？"
        elif "教育" in query or "学历" in query:
            return "关于我的教育背景，我可以分享一些相关信息。你想了解哪个阶段的教育经历？"
        else:
            return "这是一个有趣的问题。作为你的数字分身，我正在学习如何更好地回答这类问题。你能提供更多具体的信息吗？"
    
    def process_data_source(self, data_source_id: uuid.UUID) -> bool:
        """
        处理数据源，提取信息并存入向量数据库
        
        参数:
        - data_source_id: 数据源ID
        
        返回:
        - 处理是否成功
        """
        # 获取数据源
        data_source = self.db.query(TwinDataSource).filter(TwinDataSource.id == data_source_id).first()
        if not data_source:
            return False
        
        try:
            # 更新状态为处理中
            data_source.process_status = "processing"
            self.db.commit()
            
            # 处理文档内容
            chunks = self._chunk_document(data_source)
            
            # 生成摘要和洞察
            summary, insight = self._generate_summary_and_insight(data_source)
            
            # 更新数据源信息
            data_source.summary = summary
            data_source.insight = insight
            data_source.chunk_count = len(chunks)
            data_source.processed_at = datetime.now()
            data_source.process_status = "processed"
            self.db.commit()
            
            # 更新向量数据库
            self._update_vector_database(data_source_id)
            
            # 更新数字分身的Bio
            self._update_twin_bio(data_source.twin_id)
            
            return True
        except Exception as e:
            logger.error(f"Error processing data source: {str(e)}")
            data_source.process_status = "failed"
            data_source.error_message = str(e)
            self.db.commit()
            return False
    
    def _chunk_document(self, data_source: TwinDataSource) -> List[str]:
        """
        将文档切分成块
        
        参数:
        - data_source: 数据源对象
        
        返回:
        - 文档块列表
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # 获取文档内容
            content = data_source.content
            if not content:
                return []
            
            # 切分文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_text(content)
            
            # 存储块
            for i, chunk_content in enumerate(chunks):
                chunk = TwinChunk(
                    data_source_id=data_source.id,
                    content=chunk_content,
                    sequence=i,
                    metadata={"source": data_source.name, "index": i}
                )
                self.db.add(chunk)
            
            self.db.commit()
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise
    
    def _generate_summary_and_insight(self, data_source: TwinDataSource) -> tuple:
        """
        生成文档摘要和洞察
        
        参数:
        - data_source: 数据源对象
        
        返回:
        - (摘要, 洞察)元组
        """
        try:
            # 准备提示
            content = data_source.content[:5000]  # 限制内容长度
            
            # 使用豆包客户端生成摘要
            summary_prompt = [
                {"role": "system", "content": "你是一个专业的文档分析助手，擅长提取文档的主要内容并生成简洁的摘要。"},
                {"role": "user", "content": f"请为以下文档生成一个简洁的摘要（200字以内）：\n\n{content}"}
            ]
            summary_response = doubao_client.chat_completion(summary_prompt, temperature=0.3, max_tokens=300)
            summary = doubao_client.process_response(summary_response) or "无法生成摘要"
            
            # 生成洞察
            insight_prompt = [
                {"role": "system", "content": "你是一个专业的文档分析助手，擅长从文档中提取关键洞察和见解。"},
                {"role": "user", "content": f"请分析以下文档，提取3-5个关键洞察或见解：\n\n{content}"}
            ]
            insight_response = doubao_client.chat_completion(insight_prompt, temperature=0.3, max_tokens=500)
            insight = doubao_client.process_response(insight_response) or "无法生成洞察"
            
            return summary, insight
        except Exception as e:
            logger.error(f"Error generating summary and insight: {str(e)}")
            return "处理失败", "处理失败"

    def _update_vector_database(self, data_source_id: uuid.UUID) -> bool:
        """
        更新向量数据库
        
        参数:
        - data_source_id: 数据源ID
        
        返回:
        - 更新是否成功
        """
        if not self.chroma_client:
            logger.warning("ChromaDB not available, skipping vector database update")
            return False
        
        try:
            # 获取数据源及其所属的数字分身
            data_source = self.db.query(TwinDataSource).filter(TwinDataSource.id == data_source_id).first()
            if not data_source:
                return False
            
            twin = self.db.query(DigitalTwin).filter(DigitalTwin.id == data_source.twin_id).first()
            if not twin:
                return False
            
            # 创建或获取向量集合
            collection_name = twin.vector_collection_name
            if not collection_name:
                collection_name = f"twin_{str(twin.id).replace('-', '')}"
                twin.vector_collection_name = collection_name
                self.db.commit()
            
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"twin_id": str(twin.id)}
                )
            
            # 获取数据源的所有块
            chunks = self.db.query(TwinChunk).filter(TwinChunk.data_source_id == data_source_id).all()
            if not chunks:
                return True  # 没有块需要添加
            
            # 准备向量数据库文档
            from sentence_transformers import SentenceTransformer
            
            # 加载嵌入模型
            model_name = self.embedding_model
            model = SentenceTransformer(model_name)
            
            # 处理块
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            for chunk in chunks:
                # 生成嵌入
                embedding = model.encode(chunk.content)
                
                # 添加到批次
                documents.append(chunk.content)
                embeddings.append(embedding.tolist())
                ids.append(str(chunk.id))
                metadatas.append({
                    "data_source_id": str(data_source.id),
                    "data_source_name": data_source.name,
                    "chunk_sequence": chunk.sequence or 0,
                    "twin_id": str(twin.id)
                })
            
            # 添加到向量数据库
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            logger.error(f"Error updating vector database: {str(e)}")
            return False
    
    def _update_twin_bio(self, twin_id: uuid.UUID) -> bool:
        """
        更新数字分身的Bio信息
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 更新是否成功
        """
        if not self.l1_generator:
            logger.warning("L1Generator not available, skipping Bio update")
            return False
        
        try:
            # 获取数字分身
            twin = self.db.query(DigitalTwin).filter(DigitalTwin.id == twin_id).first()
            if not twin:
                return False
            
            # 获取所有数据源
            data_sources = self.db.query(TwinDataSource).filter(
                TwinDataSource.twin_id == twin_id,
                TwinDataSource.process_status == "processed"
            ).all()
            
            if not data_sources:
                return False  # 没有处理完成的数据源
            
            # 生成Bio和Shades
            bio, shades = self._generate_bio(twin, data_sources)
            
            # 更新数字分身信息
            twin.bio_content = bio.content if bio else ""
            twin.bio_summary = bio.summary if bio else ""
            twin.shades = shades
            twin.last_trained_at = datetime.now()
            twin.status = "active"
            
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating twin Bio: {str(e)}")
            return False
    
    def _generate_bio(self, twin: DigitalTwin, data_sources: List[TwinDataSource]) -> tuple:
        """
        使用Second-Me的L1Generator生成数字分身的生物信息
        
        参数:
        - twin: 数字分身对象
        - data_sources: 数据源列表
        
        返回:
        - (bio, shades): 生成的Bio对象和特征列表
        """
        if not self.l1_generator:
            logger.error("L1Generator未初始化，无法生成Bio")
            return None, None
        
        try:
            # 将数据源内容转换为Note对象
            notes = []
            for ds in data_sources:
                # 获取数据源的所有文本块
                chunks = self.db.query(TwinChunk).filter(
                    TwinChunk.data_source_id == ds.id
                ).all()
                
                # 将文本块转换为Note对象
                for i, chunk in enumerate(chunks):
                    note = Note(
                        noteId=i + 1000 * len(notes),  # 为每个Note生成唯一ID
                        content=chunk.content,
                        createTime=ds.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        memoryType="TEXT",
                        title=f"{ds.name} - Part {i+1}",
                        summary=chunk.summary if hasattr(chunk, 'summary') and chunk.summary else "",
                        insight=chunk.insight if hasattr(chunk, 'insight') and chunk.insight else ""
                    )
                    notes.append(note)
                
            if not notes:
                logger.warning(f"数字分身 {twin.id} 没有可处理的数据")
                return None, None
            
            # 初始化空Bio对象
            old_bio = Bio()
            
            # 获取或生成集群
            clusters = self.db.query(TwinCluster).filter(
                TwinCluster.twin_id == twin.id
            ).all()
            
            # 如果没有现有集群，则生成新的集群
            if not clusters:
                logger.info(f"为数字分身 {twin.id} 生成集群")
                
                # 使用TopicsGenerator生成主题（简化版实现）
                try:
                    # 调用L1Generator的generate_topics方法
                    topics_result = self.l1_generator.generate_topics(notes)
                    
                    if topics_result and "clusters" in topics_result:
                        # 将生成的集群保存到数据库
                        for cluster_data in topics_result["clusters"]:
                            cluster = TwinCluster(
                                twin_id=twin.id,
                                name=cluster_data.get("name", "未命名集群"),
                                description=cluster_data.get("description", ""),
                                notes_count=len(cluster_data.get("notes", [])),
                                center_embedding=json.dumps(cluster_data.get("center_embedding", []))
                            )
                            self.db.add(cluster)
                            
                            # 读取集群中的Note IDs
                            note_ids = [note.get("id") for note in cluster_data.get("notes", [])]
                            
                            # 后续处理...
                    else:
                        logger.warning("TopicsGenerator没有返回有效的集群数据")
                except Exception as e:
                    logger.error(f"生成主题时出错: {str(e)}")
                    # 创建一个默认集群
                    cluster = TwinCluster(
                        twin_id=twin.id,
                        name="默认集群",
                        description="自动创建的默认集群",
                        notes_count=len(notes)
                    )
                    self.db.add(cluster)
                
                self.db.commit()
                
                # 重新获取集群
                clusters = self.db.query(TwinCluster).filter(
                    TwinCluster.twin_id == twin.id
                ).all()
                
            # 将数据库中的集群转换为Second-Me的Cluster对象
            l1_clusters = []
            for cluster in clusters:
                # 获取集群中的特征
                shades = self.db.query(TwinShade).filter(
                    TwinShade.cluster_id == cluster.id
                ).all()
                
                # 创建L1 Cluster对象
                l1_cluster = Cluster(
                    clusterId=cluster.id,
                    is_new=False
                )
                
                # 将集群中的笔记添加到L1 Cluster
                # 简化实现，实际应关联真实的集群笔记
                for i, note in enumerate(notes[:min(len(notes), cluster.notes_count)]):
                    l1_cluster.add_memory(Memory(memoryId=note.id, embedding=note.embedding))
                
                l1_clusters.append(l1_cluster)
            
            # 使用L1Generator生成Bio
            try:
                # 调用L1Generator的gen_global_biography方法
                bio = self.l1_generator.gen_global_biography(old_bio, l1_clusters)
                
                if not bio:
                    logger.error("L1Generator未能生成有效的Bio")
                    return None, None
                
                # 为每个集群生成特征信息
                shades = []
                for i, cluster in enumerate(l1_clusters):
                    # 获取集群对应的笔记
                    cluster_notes = []
                    for memory in cluster.memory_list:
                        # 查找memory_id对应的Note
                        for note in notes:
                            if note.id == memory.memory_id:
                                cluster_notes.append(note)
                                break
                
                    # 已有的特征信息
                    existing_shades = []
                    db_shades = self.db.query(TwinShade).filter(
                        TwinShade.cluster_id == cluster.cluster_id
                    ).all()
                    
                    for db_shade in db_shades:
                        existing_shades.append(ShadeInfo(
                            id=db_shade.id,
                            name=db_shade.name,
                            aspect=db_shade.aspect,
                            icon=db_shade.icon,
                            descThirdView=db_shade.description_third_view,
                            contentThirdView=db_shade.content_third_view,
                            descSecondView=db_shade.description_second_view,
                            contentSecondView=db_shade.content_second_view
                        ))
                    
                    # 生成新的特征信息
                    try:
                        shade_info = self.l1_generator.gen_shade_for_cluster(
                            old_memory_list=[],
                            new_memory_list=cluster_notes,
                            shade_info_list=existing_shades
                        )
                        
                        if shade_info:
                            shades.append(shade_info)
                    except Exception as e:
                        logger.error(f"为集群 {cluster.cluster_id} 生成特征时出错: {str(e)}")
                
                return bio, shades
                
            except Exception as e:
                logger.error(f"生成Bio时出错: {str(e)}")
                return None, None
            
        except Exception as e:
            logger.error(f"_generate_bio方法执行出错: {str(e)}")
            return None, None

    def generate_clusters(self, twin_id: uuid.UUID) -> bool:
        """
        生成数字分身的聚类
        
        参数:
        - twin_id: 数字分身ID
        
        返回:
        - 生成是否成功
        """
        try:
            # 获取数字分身
            twin = self.db.query(DigitalTwin).filter(DigitalTwin.id == twin_id).first()
            if not twin:
                return False
            
            # 获取所有处理完成的数据源
            data_sources = self.db.query(TwinDataSource).filter(
                TwinDataSource.twin_id == twin_id,
                TwinDataSource.process_status == "processed"
            ).all()
            
            if not data_sources:
                return False  # 没有处理完成的数据源
            
            # 获取所有数据块
            chunks = []
            for ds in data_sources:
                ds_chunks = self.db.query(TwinChunk).filter(TwinChunk.data_source_id == ds.id).all()
                chunks.extend(ds_chunks)
            
            if not chunks:
                return False  # 没有数据块
            
            # 准备嵌入向量
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
            from sentence_transformers import SentenceTransformer
            
            # 加载嵌入模型
            model_name = self.embedding_model
            model = SentenceTransformer(model_name)
            
            # 提取文本和生成嵌入
            texts = [chunk.content for chunk in chunks]
            embeddings = model.encode(texts)
            
            # 聚类
            n_clusters = min(5, len(chunks))  # 最多5个聚类
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # 为每个聚类创建TwinCluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    # 创建新聚类
                    cluster = TwinCluster(
                        twin_id=twin_id,
                        name=f"Cluster {label+1}",
                        memory_ids=[]
                    )
                    self.db.add(cluster)
                    self.db.flush()  # 获取ID
                    clusters[label] = cluster
                
                # 添加块ID到聚类
                clusters[label].memory_ids.append(str(chunks[i].id))
            
            self.db.commit()
            
            # 为每个聚类生成Shade
            for label, cluster in clusters.items():
                # 获取聚类中的块
                cluster_chunks = [chunks[i] for i, l in enumerate(cluster_labels) if l == label]
                
                # 计算聚类中心
                cluster_center = np.mean([embeddings[i] for i, l in enumerate(cluster_labels) if l == label], axis=0)
                cluster.center_embedding = cluster_center.tolist()
                
                # 生成聚类摘要，用作Shade名称
                chunk_texts = [c.content[:200] for c in cluster_chunks[:5]]  # 限制数量和长度
                combined_text = "\n\n".join(chunk_texts)
                
                summary_prompt = [
                    {"role": "system", "content": "你是一个专业的文本聚类分析助手，擅长为一组相关文本生成简短的主题描述。"},
                    {"role": "user", "content": f"为以下一组相关文本生成一个简短的主题名称（5个字以内）：\n\n{combined_text}"}
                ]
                summary_response = doubao_client.chat_completion(summary_prompt, temperature=0.3, max_tokens=50)
                topic_name = doubao_client.process_response(summary_response) or f"主题 {label+1}"
                
                # 生成Shade描述
                description_prompt = [
                    {"role": "system", "content": "你是一个专业的文本聚类分析助手，擅长为一组相关文本生成详细的描述。"},
                    {"role": "user", "content": f"为以下一组相关文本生成一个详细的描述（200字以内）：\n\n{combined_text}"}
                ]
                description_response = doubao_client.chat_completion(description_prompt, temperature=0.4, max_tokens=300)
                description = doubao_client.process_response(description_response) or f"主题 {label+1} 的描述"
                
                # 创建Shade
                shade = TwinShade(
                    cluster_id=cluster.id,
                    twin_id=twin_id,
                    name=topic_name,
                    aspect=f"topic_{label+1}",
                    desc_third_view=f"关于{topic_name}的描述",
                    content_third_view=description,
                    desc_second_view=f"你的{topic_name}",
                    content_second_view=description,
                    confidence_level="MEDIUM"
                )
                self.db.add(shade)
            
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error generating clusters: {str(e)}")
            return False


# 全局函数，用于在服务中调用
def generate_twin_response(db: Session, twin_id: uuid.UUID, query: str, conversation_id: uuid.UUID) -> str:
    """全局函数：生成数字分身回复"""
    engine = TwinEngine(db)
    return engine.generate_twin_response(twin_id, query, conversation_id)


def process_data_source(db: Session, data_source_id: uuid.UUID) -> bool:
    """全局函数：处理数据源"""
    engine = TwinEngine(db)
    return engine.process_data_source(data_source_id)


def generate_clusters(db: Session, twin_id: uuid.UUID) -> bool:
    """全局函数：生成聚类"""
    engine = TwinEngine(db)
    return engine.generate_clusters(twin_id)


# 向量数据库辅助函数
def create_chroma_collection(twin_id: uuid.UUID) -> Any:
    """创建向量数据库集合"""
    import chromadb
    
    collection_name = f"twin_{str(twin_id).replace('-', '')}"
    
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        try:
            collection = client.get_collection(name=collection_name)
        except:
            collection = client.create_collection(
                name=collection_name,
                metadata={"twin_id": str(twin_id)}
            )
        return collection
    except Exception as e:
        logger.error(f"Error creating Chroma collection: {str(e)}")
        return None


def add_documents_to_chroma(collection: Any, documents: List[Dict[str, Any]]) -> List[str]:
    """向向量数据库添加文档"""
    if not collection:
        return []
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 加载嵌入模型
        model_name = settings.EMBEDDING_MODEL
        model = SentenceTransformer(model_name)
        
        # 准备数据
        texts = [doc["content"] for doc in documents]
        embeddings = model.encode(texts).tolist()
        ids = [doc["id"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # 添加到向量数据库
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        return ids
    except Exception as e:
        logger.error(f"Error adding documents to Chroma: {str(e)}")
        return []


def query_chroma(collection: Any, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """查询向量数据库"""
    if not collection:
        return []
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 加载嵌入模型
        model_name = settings.EMBEDDING_MODEL
        model = SentenceTransformer(model_name)
        
        # 生成查询嵌入
        query_embedding = model.encode(query).tolist()
        
        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 处理结果
        response = []
        if results and "documents" in results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                response.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {},
                    "id": results["ids"][0][i] if "ids" in results and results["ids"] else None,
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
        
        return response
    except Exception as e:
        logger.error(f"Error querying Chroma: {str(e)}")
        return [] 
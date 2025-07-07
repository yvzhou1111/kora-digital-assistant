import { useContext } from 'react';
import { TwinContext } from '@/contexts/TwinContext';
import { DigitalTwin, KnowledgeSource, KnowledgeSourceType } from '@/types/twin';

/**
 * Custom hook for digital twin functionality
 * @returns Digital twin state and methods
 */
export const useTwin = () => {
  const context = useContext(TwinContext);
  
  if (context === undefined) {
    throw new Error('useTwin must be used within a TwinProvider');
  }
  
  /**
   * Create a new digital twin
   */
  const createTwin = async (data: Partial<DigitalTwin>) => {
    try {
      const twin = await context.createTwin(data);
      return { success: true, twin };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '创建数字分身失败'
      };
    }
  };
  
  /**
   * Update an existing digital twin
   */
  const updateTwin = async (id: string, data: Partial<DigitalTwin>) => {
    try {
      const twin = await context.updateTwin(id, data);
      return { success: true, twin };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '更新数字分身失败'
      };
    }
  };
  
  /**
   * Delete a digital twin
   */
  const deleteTwin = async (id: string) => {
    try {
      await context.deleteTwin(id);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '删除数字分身失败'
      };
    }
  };
  
  /**
   * Add a knowledge source to a twin
   */
  const addKnowledgeSource = async (
    twinId: string, 
    sourceData: {
      type: KnowledgeSourceType;
      name: string;
      content?: string;
      url?: string;
    }
  ) => {
    try {
      const source = await context.addKnowledgeSource(twinId, sourceData);
      return { success: true, source };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '添加知识源失败'
      };
    }
  };
  
  /**
   * Create a new conversation with a twin
   */
  const startConversation = async (twinId: string, title: string) => {
    try {
      const conversation = await context.createConversation(twinId, title);
      return { success: true, conversation };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '创建对话失败'
      };
    }
  };
  
  /**
   * Send a message to a twin in a conversation
   */
  const sendMessage = async (conversationId: string, content: string) => {
    try {
      await context.sendMessage(conversationId, content);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '发送消息失败'
      };
    }
  };
  
  /**
   * Get twin skill level label
   */
  const getSkillLevelLabel = (level: number) => {
    switch (level) {
      case 1: return '入门';
      case 2: return '初级';
      case 3: return '中级';
      case 4: return '高级';
      case 5: return '专家';
      default: return '未知';
    }
  };
  
  /**
   * Check if a twin has any skill in a category
   */
  const hasSkillsInCategory = (category: string) => {
    return context.skills.some(skill => skill.category === category);
  };
  
  return {
    twins: context.twins,
    currentTwin: context.currentTwin,
    conversations: context.conversations,
    currentConversation: context.currentConversation,
    knowledgeSources: context.knowledgeSources,
    skills: context.skills,
    isLoading: context.isLoading,
    error: context.error,
    setCurrentTwin: context.setCurrentTwin,
    createTwin,
    updateTwin,
    deleteTwin,
    addKnowledgeSource,
    startConversation,
    sendMessage,
    getSkillLevelLabel,
    hasSkillsInCategory,
  };
}; 
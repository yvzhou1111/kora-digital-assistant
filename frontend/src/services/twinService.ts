import api from './api';
import { 
  DigitalTwin, 
  KnowledgeSource, 
  TwinConversation, 
  TwinMessage,
  TwinSkill,
  KnowledgeSourceType
} from '@/types/twin';

const twinService = {
  // Get all user's twins
  getAllTwins: async (): Promise<DigitalTwin[]> => {
    try {
      const response = await api.get<DigitalTwin[]>('/twins');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get a specific twin by ID
  getTwinById: async (id: string): Promise<DigitalTwin> => {
    try {
      const response = await api.get<DigitalTwin>(`/twins/${id}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Create a new digital twin
  createTwin: async (twinData: Partial<DigitalTwin>): Promise<DigitalTwin> => {
    try {
      const response = await api.post<DigitalTwin>('/twins', twinData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Update an existing twin
  updateTwin: async (id: string, twinData: Partial<DigitalTwin>): Promise<DigitalTwin> => {
    try {
      const response = await api.put<DigitalTwin>(`/twins/${id}`, twinData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Delete a twin
  deleteTwin: async (id: string): Promise<void> => {
    try {
      await api.delete(`/twins/${id}`);
    } catch (error) {
      throw error;
    }
  },

  // Get twin's knowledge sources
  getKnowledgeSources: async (twinId: string): Promise<KnowledgeSource[]> => {
    try {
      const response = await api.get<KnowledgeSource[]>(`/twins/${twinId}/knowledge-sources`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Add a knowledge source to a twin
  addKnowledgeSource: async (
    twinId: string, 
    source: { 
      type: KnowledgeSourceType, 
      name: string, 
      content?: string, 
      url?: string 
    }
  ): Promise<KnowledgeSource> => {
    try {
      const response = await api.post<KnowledgeSource>(`/twins/${twinId}/knowledge-sources`, source);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Upload a file as knowledge source
  uploadKnowledgeFile: async (
    twinId: string,
    file: File,
    type: KnowledgeSourceType,
    name: string
  ): Promise<KnowledgeSource> => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', type);
      formData.append('name', name);

      const response = await api.post<KnowledgeSource>(
        `/twins/${twinId}/knowledge-sources/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Delete a knowledge source
  deleteKnowledgeSource: async (twinId: string, sourceId: string): Promise<void> => {
    try {
      await api.delete(`/twins/${twinId}/knowledge-sources/${sourceId}`);
    } catch (error) {
      throw error;
    }
  },

  // Get twin's conversations
  getConversations: async (twinId: string): Promise<TwinConversation[]> => {
    try {
      const response = await api.get<TwinConversation[]>(`/twins/${twinId}/conversations`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get a specific conversation by ID
  getConversation: async (twinId: string, conversationId: string): Promise<TwinConversation> => {
    try {
      const response = await api.get<TwinConversation>(`/twins/${twinId}/conversations/${conversationId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Create a new conversation
  createConversation: async (twinId: string, title: string): Promise<TwinConversation> => {
    try {
      const response = await api.post<TwinConversation>(`/twins/${twinId}/conversations`, { title });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Send a message in a conversation
  sendMessage: async (
    twinId: string,
    conversationId: string,
    content: string
  ): Promise<TwinMessage> => {
    try {
      const response = await api.post<TwinMessage>(
        `/twins/${twinId}/conversations/${conversationId}/messages`,
        { content }
      );
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get twin's skills
  getTwinSkills: async (twinId: string): Promise<TwinSkill[]> => {
    try {
      const response = await api.get<TwinSkill[]>(`/twins/${twinId}/skills`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get twin test performance
  testTwin: async (twinId: string, questions: string[]): Promise<{
    score: number;
    responses: { question: string; answer: string; score: number }[];
  }> => {
    try {
      const response = await api.post(`/twins/${twinId}/test`, { questions });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Train/retrain a twin
  trainTwin: async (twinId: string): Promise<{ status: string; message: string }> => {
    try {
      const response = await api.post(`/twins/${twinId}/train`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};

export default twinService; 